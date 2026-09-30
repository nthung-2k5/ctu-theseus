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
import re
import shutil
import uuid
from typing import Any

import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.backends.base import CustomModelKind
from theseus.backends.registry import find_backend, list_backends
from theseus.db.models import CustomModel, TrainingRun, User
from theseus.jobs.dispatcher import nudge
from theseus.schemas.custom_models import (
    CreateCustomModelBody,
    CustomModelKindOut,
    CustomModelOut,
    UpdateCustomModelBody,
    UploadUrlBody,
)
from theseus.services import custom_models, storage
from theseus.services.task_registry import TASK_REGISTRY
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

# A Hugging Face repo id ("org/name", or a bare legacy name) or a timm name ("resnet50.a1_in1k").
_SOURCE_REF = re.compile(r"^[A-Za-z0-9][\w.\-]*(/[\w.\-]+)?$")
_UPLOAD_EXTENSIONS = {".zip": "bundle.zip", ".safetensors": "model.safetensors"}
_UPLOAD_URL_TTL_SECONDS = 3600
_VALIDATION_ATTEMPTS = 3


# -- Reading -----------------------------------------------------------------------------------


def _out(row: CustomModel, run_count: int, owner_email: str | None = None) -> CustomModelOut:
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
        tasks=list(row.tasks or []),
        status=row.status,
        last_error=row.last_error,
        enabled=row.enabled,
        archived=row.archived_at is not None,
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
                        tasks=[],
                    ),
                )
                entry.tasks.append(task.id)
    return list(kinds.values())


# -- Validation helpers ------------------------------------------------------------------------


def _kind_for_tasks(backend_id: str, kind_id: str, tasks: list[str]) -> tuple[Any, CustomModelKind]:
    """The backend and the kind, checked to exist, be usable, and be offered for EVERY task listed."""
    backend = find_backend(backend_id)
    if backend is None:
        raise HTTPException(400, f"Unknown trainer backend '{backend_id}'")
    if (reason := backend.available()) is not None:
        raise HTTPException(400, f"Trainer backend '{backend_id}' is not available: {reason}")
    found: CustomModelKind | None = None
    for task_id in tasks:
        task = TASK_REGISTRY.get(task_id)
        if task is None:
            raise HTTPException(400, f"Unknown task '{task_id}'")
        match = next((k for k in backend.custom_model_kinds(task) if k.id == kind_id), None)
        if match is None:
            raise HTTPException(400, f"'{backend_id}' has no '{kind_id}' custom model for {task.label}")
        found = match
    assert found is not None  # tasks is non-empty (schema min_length)
    if found.unavailable_reason:
        raise HTTPException(400, found.unavailable_reason)
    return backend, found


# -- Writing -----------------------------------------------------------------------------------


async def create(
    session: AsyncSession, body: CreateCustomModelBody, actor_id: uuid.UUID, *, global_: bool
) -> CustomModel:
    tasks = list(dict.fromkeys(body.tasks))
    _, kind = _kind_for_tasks(body.backend, body.kind, tasks)
    if body.source_kind not in kind.source_kinds:
        raise HTTPException(400, f"A '{kind.id}' model cannot come from source '{body.source_kind}'")

    source_ref = (body.source_ref or "").strip() or None
    if body.source_kind == "hub":
        if source_ref is None or not _SOURCE_REF.match(source_ref):
            raise HTTPException(
                400, "Enter the model's Hugging Face repository id (e.g. bert-base-uncased or org/name)"
            )
    elif source_ref is not None:
        raise HTTPException(400, "An uploaded model has no source reference")

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
        tasks=tasks,
        # A Hub model needs nothing more from the user, so it goes straight to validation.
        status="uploaded" if body.source_kind == "hub" else "pending_upload",
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
    if body.tasks is not None:
        tasks = list(dict.fromkeys(body.tasks))
        _kind_for_tasks(row.backend, row.kind, tasks)
        row.tasks = tasks
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
    if row.source_kind != "upload" or row.status != "pending_upload":
        raise HTTPException(409, "This model is not waiting for an upload")
    limit = get_settings().max_custom_model_bytes
    if body.size_bytes > limit:
        raise HTTPException(413, f"The file is larger than the {limit // 2**20} MiB limit")
    ext = next((e for e in _UPLOAD_EXTENSIONS if body.filename.lower().endswith(e)), None)
    if ext is None:
        raise HTTPException(400, "Upload a .zip of the model folder, or a single .safetensors file")

    key = custom_models.custom_model_key(row.id, _UPLOAD_EXTENSIONS[ext])
    row.storage_key = key
    await session.commit()
    content_type = "application/zip" if ext == ".zip" else "application/octet-stream"
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
