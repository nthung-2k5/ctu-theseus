"""Create, edit, upload, retry and delete custom models: the rules, shared by the user routes
(/api/models, owner-scoped) and the admin routes (/api/admin/models, everyone's).

Who may touch what is decided here and nowhere else: `load` reports a model that is not yours exactly like
one that does not exist (404), so an id cannot be used to find out whether someone else has a model.

Everything that says WHAT a model is (backend, kind, source, revision, checksum) is fixed at creation.
Editing changes only the name, description, task list and the enabled switch, so a training run's record
of which model it used never changes meaning underneath it.
"""

import asyncio
import logging
import shutil
import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain.core import constants as C
from omnitrain.db.models import CustomModel, TrainingRun, User
from omnitrain.domain.task_registry import TASK_REGISTRY
from omnitrain.jobs.dispatcher import nudge
from omnitrain.plugins.backends.base import CustomModelKind
from omnitrain.plugins.backends.registry import find_backend, list_backends
from omnitrain.plugins.model_sources import SourceRefError, find_source, get_source
from omnitrain.schemas.custom_models import (
    CreateCustomModelBody,
    CustomModelKindOut,
    CustomModelOut,
    InspectModelBody,
    InspectModelResponse,
    UpdateCustomModelBody,
    UploadUrlBody,
)
from omnitrain.services import custom_models, storage
from omnitrain.services.model_catalog import kind_and_tasks
from omnitrain.settings import get_settings

logger = logging.getLogger(__name__)

# A Hugging Face repo id ("org/name", or a bare legacy name) or a timm name ("resnet50.a1_in1k").
_UPLOAD_FILENAME = "bundle.zip"
_UPLOAD_URL_TTL_SECONDS = 3600
_VALIDATION_ATTEMPTS = 3


# -- Reading -----------------------------------------------------------------------------------


def _out(row: CustomModel, run_count: int, owner_email: str | None = None) -> CustomModelOut:
    backend = find_backend(row.backend)
    return CustomModelOut(
        id=row.id,
        name=row.name,
        description=row.description,
        backend=row.backend,
        kind=row.kind,
        source_kind=row.source_kind,  # type: ignore[arg-type]
        source_ref=row.source_ref,
        revision=row.revision,
        size_bytes=row.size_bytes,
        tasks=kind_and_tasks(backend, row.kind)[1] if backend else [],
        status=row.status,
        last_error=row.last_error,
        enabled=row.enabled,
        archived=row.archived_at is not None,
        private=bool((row.spec or {}).get("private")),
        scope="private" if row.owner_user_id is not None else "global",
        owner_user_id=row.owner_user_id,
        owner_email=owner_email,
        run_count=run_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _run_count() -> sa.ScalarSelect:
    return (
        sa.select(sa.func.count()).select_from(TrainingRun).where(TrainingRun.custom_model_id == CustomModel.id)
    ).scalar_subquery()


async def load(session: AsyncSession, model_id: uuid.UUID, actor_id: uuid.UUID, *, is_admin: bool) -> CustomModel:
    row = await session.get(CustomModel, model_id)
    if row is None or (not is_admin and row.owner_user_id != actor_id):
        raise HTTPException(404, "Model not found")
    return row


def select_count(model_id: uuid.UUID) -> sa.Select:
    """How many training runs used this model."""
    return sa.select(sa.func.count()).select_from(TrainingRun).where(TrainingRun.custom_model_id == model_id)


async def describe(session: AsyncSession, row: CustomModel, *, with_owner: bool) -> CustomModelOut:
    runs = (await session.execute(select_count(row.id))).scalar_one()
    email = None
    if with_owner and row.owner_user_id is not None:
        email = (await session.execute(sa.select(User.email).where(User.id == row.owner_user_id))).scalar_one_or_none()
    return _out(row, runs, email)


async def list_models(
    session: AsyncSession,
    *,
    owner_user_id: uuid.UUID | None = None,
    scope: str | None = None,
    status: str | None = None,
    q: str | None = None,
    include_archived: bool = False,
    with_owner: bool = False,
) -> list[CustomModelOut]:
    """`owner_user_id` restricts to one user's own models (the user routes). Admins leave it unset and may
    filter by `scope` (global / private), `status` and a name search."""
    stmt = sa.select(CustomModel, _run_count(), User.email).outerjoin(User, User.id == CustomModel.owner_user_id)
    if owner_user_id is not None:
        stmt = stmt.where(CustomModel.owner_user_id == owner_user_id)
    if scope == "global":
        stmt = stmt.where(CustomModel.owner_user_id.is_(None))
    elif scope == "private":
        stmt = stmt.where(CustomModel.owner_user_id.is_not(None))
    if status:
        stmt = stmt.where(CustomModel.status == status)
    if q:
        needle = q.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        stmt = stmt.where(sa.func.lower(CustomModel.name).like(f"%{needle}%", escape="\\"))
    if not include_archived:
        stmt = stmt.where(CustomModel.archived_at.is_(None))
    rows = (await session.execute(stmt.order_by(CustomModel.created_at.desc(), CustomModel.id))).all()
    return [_out(row, runs, email if with_owner else None) for row, runs, email in rows]


def list_kinds() -> list[CustomModelKindOut]:
    """Every custom model kind an available backend accepts, with the tasks each can be used for."""
    kinds: dict[tuple[str, str], CustomModelKindOut] = {}
    for backend in list_backends():
        if backend.available() is not None:
            continue
        for task in TASK_REGISTRY.values():
            if not backend.supports(task):
                continue
            for k in backend.custom_model_kinds(task):
                entry = kinds.setdefault(
                    (backend.id, k.id),
                    CustomModelKindOut(
                        backend=backend.id,
                        backend_label=backend.label,
                        id=k.id,
                        label=k.label,
                        description=k.description,
                        modality=k.modality,
                        source_kinds=k.source_kinds,
                        status=k.status,
                        unavailable_reason=k.unavailable_reason,
                        needs_files=k.materialize,
                        tasks=[],
                    ),
                )
                entry.tasks.append(task.id)
    return list(kinds.values())


def _suggest(config: dict[str, Any]) -> tuple[str, str] | None:
    """(backend, kind) the config's architectures point to, among kinds usable right now."""
    for backend in list_backends():
        if backend.available() is not None:
            continue
        kind_id = backend.suggest_custom_kind(config)
        if kind_id is None:
            continue
        kind, _ = kind_and_tasks(backend, kind_id)
        if kind is not None and kind.unavailable_reason is None:
            return backend.id, kind_id
    return None


def _new_inspection(source_ref: str | None) -> InspectModelResponse:
    return InspectModelResponse(
        source_ref=source_ref,
        revision=None,
        size_bytes=None,
        model_type=None,
        architectures=[],
        suggested_backend=None,
        suggested_kind=None,
        errors=[],
        warnings=[],
    )


def _describe_config(out: InspectModelResponse, config: dict[str, Any]) -> None:
    """Fill in what a config.json says about the model: its family, and the kind it points to."""
    architectures = [a for a in config.get("architectures") or [] if isinstance(a, str)]
    out.model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
    out.architectures = architectures
    if "auto_map" in config:
        out.errors.append("Models that need custom code (auto_map / trust_remote_code) are not supported")
    if suggestion := _suggest(config):
        out.suggested_backend, out.suggested_kind = suggestion
    else:
        listed = ", ".join(architectures) or "none listed"
        out.warnings.append(
            f"config.json does not say what task this model is for (architectures: {listed}). "
            "Choose the kind of model yourself."
        )


def _inspect_hub_sync(repo_id: str, token: str | None) -> InspectModelResponse:
    out = _new_inspection(repo_id)
    try:
        pin = custom_models.hub_pin(repo_id, token)
        out.revision, out.size_bytes, out.private = pin.revision, pin.size_bytes, pin.private
        limit = get_settings().max_custom_model_bytes
        if pin.size_bytes > limit:
            out.errors.append(f"The model is {pin.size_bytes // 2**20} MiB, over the {limit // 2**20} MiB limit")
        if not pin.has_safetensors:
            out.errors.append("The repository has no .safetensors weights (pickle-based weights are not accepted)")
        config = custom_models.fetch_hub_config(repo_id, pin.revision, token) if pin.has_config else None
    except custom_models.CustomModelFileError as e:
        out.errors.append(str(e))
        return out

    if config is None:
        out.warnings.append(
            "No config.json was found, so the task cannot be read from it. Choose the kind of model yourself."
        )
        return out
    _describe_config(out, config)
    return out


async def inspect_model(body: InspectModelBody, token: str | None = None) -> InspectModelResponse:
    """Check a model before importing it, creating nothing: look a Hub repository up and read its config.json,
    or describe a config.json the browser read out of an uploaded zip.

    `token` is the caller's own Hugging Face token, so they can look at their private and gated repositories.
    None for a model made for everyone, which never carries one person's access."""
    if body.config is not None:
        out = _new_inspection(None)
        _describe_config(out, body.config)
        return out
    try:
        source_ref = get_source("hub").normalize_ref(body.source_ref)
    except SourceRefError as e:
        raise HTTPException(400, str(e)) from None
    assert source_ref is not None
    return await asyncio.to_thread(_inspect_hub_sync, source_ref, token)


# -- Validation helpers ------------------------------------------------------------------------


def _kind_for(backend_id: str, kind_id: str) -> tuple[Any, CustomModelKind]:
    """The backend and the kind, checked to exist, be usable, and serve at least one task."""
    backend = find_backend(backend_id)
    if backend is None:
        raise HTTPException(400, f"Unknown trainer backend '{backend_id}'")
    if (reason := backend.available()) is not None:
        raise HTTPException(400, f"Trainer backend '{backend_id}' is not available: {reason}")
    found, _ = kind_and_tasks(backend, kind_id)
    if found is None:
        raise HTTPException(400, f"'{backend_id}' has no '{kind_id}' custom model")
    if found.unavailable_reason:
        raise HTTPException(400, found.unavailable_reason)
    return backend, found


# -- Writing -----------------------------------------------------------------------------------


async def create(
    session: AsyncSession, body: CreateCustomModelBody, actor_id: uuid.UUID, *, global_: bool
) -> CustomModel:
    _, kind = _kind_for(body.backend, body.kind)
    source = find_source(body.source_kind)
    if source is None or body.source_kind not in kind.source_kinds:
        raise HTTPException(400, f"A '{kind.id}' model cannot come from source '{body.source_kind}'")
    try:
        source_ref = source.normalize_ref(body.source_ref)
    except SourceRefError as e:
        raise HTTPException(400, str(e)) from None

    row = CustomModel(
        owner_user_id=None if global_ else actor_id,
        created_by=actor_id,
        backend=body.backend,
        kind=body.kind,
        name=body.name.strip(),
        description=body.description.strip(),
        source_kind=body.source_kind,
        source_ref=source_ref,
        spec={"materialize": kind.materialize},
        # A source that needs no upload (a Hub reference) goes straight to validation.
        status="pending_upload" if source.needs_upload else "uploaded",
        max_attempts=_VALIDATION_ATTEMPTS,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    if row.status == "uploaded":
        nudge("validate")
    return row


async def update(session: AsyncSession, row: CustomModel, body: UpdateCustomModelBody) -> CustomModel:
    if row.archived_at is not None:
        raise HTTPException(409, "This model was archived and can no longer be edited")
    if body.name is not None:
        row.name = body.name.strip()
    if body.description is not None:
        row.description = body.description.strip()
    if body.enabled is not None:
        row.enabled = body.enabled
    await session.commit()
    await session.refresh(row)
    return row


async def upload_url(session: AsyncSession, row: CustomModel, body: UploadUrlBody) -> dict[str, Any]:
    source = find_source(row.source_kind)
    if source is None or not source.needs_upload or row.status != "pending_upload":
        raise HTTPException(409, "This model is not waiting for an upload")
    limit = get_settings().max_custom_model_bytes
    if body.size_bytes > limit:
        raise HTTPException(413, f"The file is larger than the {limit // 2**20} MiB limit")
    name = body.filename.lower()
    if not name.endswith(".zip"):
        if name.endswith((".pth", ".safetensors")):
            raise HTTPException(400, "Zip the weights together with the model's config.json and upload the .zip")
        raise HTTPException(400, "Upload a .zip of the model folder")

    key = custom_models.custom_model_key(row.id, _UPLOAD_FILENAME)
    row.storage_key = key
    await session.commit()
    content_type = "application/zip"
    url = await asyncio.to_thread(storage.get_upload_url, C.BUCKET_MODELS, key, _UPLOAD_URL_TTL_SECONDS, content_type)
    return {"url": url, "key": key, "headers": {"Content-Type": content_type}, "max_bytes": limit}


async def finalize_upload(session: AsyncSession, row: CustomModel) -> CustomModel:
    """The browser says it has finished uploading: check the object is really there, then queue validation."""
    if not row.storage_key:
        raise HTTPException(409, "Request an upload URL first")
    if not await asyncio.to_thread(storage.file_exists, C.BUCKET_MODELS, row.storage_key):
        raise HTTPException(409, "The uploaded file was not found. Upload it again.")
    claimed = await session.execute(
        sa.update(CustomModel)
        .where(CustomModel.id == row.id, CustomModel.status == "pending_upload", CustomModel.storage_key.is_not(None))
        .values(status="uploaded", available_at=sa.func.now(), last_error=None)
        .returning(CustomModel.id)
    )
    if claimed.first() is None:
        raise HTTPException(409, "This model is not waiting for an upload")
    await session.commit()
    await session.refresh(row)
    nudge("validate")
    return row


async def retry(session: AsyncSession, row: CustomModel) -> CustomModel:
    """Validate a failed model again (say the Hub was down, or the user fixed the repo)."""
    claimed = await session.execute(
        sa.update(CustomModel)
        .where(CustomModel.id == row.id, CustomModel.status == "failed")
        .values(status="uploaded", attempt=0, available_at=sa.func.now(), last_error=None)
        .returning(CustomModel.id)
    )
    if claimed.first() is None:
        raise HTTPException(409, "Only a model whose validation failed can be retried")
    await session.commit()
    await session.refresh(row)
    nudge("validate")
    return row


async def delete(session: AsyncSession, row: CustomModel) -> tuple[bool, bool]:
    """Returns (deleted, archived). A model any run trained on is archived, never deleted, so those runs
    keep loading and exporting; its files stay too."""
    if row.status == "validating":
        raise HTTPException(409, "This model is being validated right now. Try again in a moment.")
    in_use = (await session.execute(select_count(row.id))).scalar_one() > 0
    if in_use:
        row.archived_at = sa.func.now()
        row.enabled = False
        await session.commit()
        return False, True

    model_id, base = row.id, custom_models.local_dir_for(row).parent
    await session.delete(row)
    await session.commit()
    # Best effort, and it must stay that way: the row is already deleted and committed, so an unreachable
    # object store must not turn a deletion that happened into an error. A leftover file is only wasted space.
    try:
        await asyncio.to_thread(storage.delete_prefix, C.BUCKET_MODELS, f"custom/{model_id}/")
    except Exception:
        logger.warning("Could not remove the stored files of deleted custom model %s", model_id, exc_info=True)
    await asyncio.to_thread(shutil.rmtree, base, True)
    return True, False
