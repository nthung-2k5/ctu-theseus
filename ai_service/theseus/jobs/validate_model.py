"""Validate a bring-your-own model: pin it, fetch it, check it, then mark it `ready`.

A custom model row is created `uploaded` (a Hub reference, or an upload the browser has finished) and
claimed off the `custom_models` table like any other job (see jobs/queue.py). The job:

  1. Hub: look the repo up, refuse a private/gated or oversized one, and PIN the commit sha as the
     model's revision. Upload: download the bundle once, hash it and unpack it (`import_upload`).
  2. Generic safety checks on the files (`custom_models.check_tree`): no pickles or code, no symlinks,
     within the size limit.
  3. The backend's own check (`TrainerBackend.validate_custom_model`) for every task the model is offered
     for: is this really a causal LM, does it have a config.json, and so on.

A problem WITH THE MODEL (bad files, missing repo, wrong architecture) fails it immediately with a
message the user can act on: retrying a deterministic failure only wastes time. Anything else (a network
hiccup talking to the Hub or S3) propagates to the dispatcher, which retries with a delay and fails the
model once attempts run out.
"""

import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from theseus.backends.base import ConfigError, CustomModelKind
from theseus.backends.registry import get_backend
from theseus.db.base import get_sessionmaker
from theseus.db.models import CustomModel
from theseus.jobs import queue
from theseus.jobs.executors import run_in_executor, validate_executor
from theseus.services import custom_models
from theseus.services.custom_models import CustomModelFileError
from theseus.services.task_registry import get_task_descriptor
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.validate_model")


@dataclass
class Validated:
    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None


def _hub_pin(row: CustomModel) -> tuple[str, int]:
    """(commit sha, bytes to download) for a Hub model, refusing one that cannot be used."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    if not row.source_ref:
        raise CustomModelFileError("A Hub model needs its repository id")
    try:
        info = HfApi().model_info(row.source_ref, files_metadata=True)
    except RepositoryNotFoundError:
        raise CustomModelFileError(f"'{row.source_ref}' was not found on the Hugging Face Hub") from None
    except GatedRepoError:
        raise CustomModelFileError(f"'{row.source_ref}' is gated: only public, ungated models are supported") from None
    except RevisionNotFoundError:
        raise CustomModelFileError(f"'{row.source_ref}' has no default revision") from None
    if info.private:
        raise CustomModelFileError(f"'{row.source_ref}' is private: only public models are supported")
    if not info.sha:
        raise CustomModelFileError(f"Could not determine the current revision of '{row.source_ref}'")

    forbidden = custom_models.FORBIDDEN_EXTENSIONS
    wanted = [
        s
        for s in info.siblings or []
        if not any(s.rfilename.lower().endswith(ext) for ext in forbidden)
        and s.rfilename.lower().endswith((".safetensors", ".json", ".txt", ".model", ".tiktoken", ".vocab", ".md"))
    ]
    return info.sha, sum(s.size or 0 for s in wanted)


def _kind_for(row: CustomModel) -> CustomModelKind:
    backend = get_backend(row.backend)
    for task_id in row.tasks:
        for kind in backend.custom_model_kinds(get_task_descriptor(task_id)):
            if kind.id == row.kind:
                return kind
    raise ConfigError(f"'{row.backend}' does not offer a '{row.kind}' model for {', '.join(row.tasks) or 'any task'}")


def validate_sync(row: CustomModel) -> Validated:
    """Everything that touches the network or the disk. Runs in a worker thread; mutates the detached
    `row` only to carry the pin forward, and returns what to persist."""
    limit = get_settings().max_custom_model_bytes
    backend = get_backend(row.backend)
    kind = _kind_for(row)
    if kind.unavailable_reason:
        raise ConfigError(kind.unavailable_reason)
    if not row.tasks:
        raise ConfigError("The model is not enabled for any task")

    result = Validated()
    if not kind.materialize:
        pass  # the framework fetches its own weights by name; nothing to pin or check on disk
    elif row.source_kind == "hub":
        revision, size = _hub_pin(row)
        if size > limit:
            raise CustomModelFileError(f"The model is {size // 2**20} MiB, over the {limit // 2**20} MiB limit")
        row.revision = result.revision = revision
        path = custom_models.ensure_local(row)
        result.size_bytes = custom_models.check_tree(path, max_bytes=limit)
    else:
        imported = custom_models.import_upload(row)
        row.sha256 = result.sha256 = imported.sha256
        result.size_bytes = custom_models.check_tree(imported.path, max_bytes=limit)

    ref = custom_models.ref_for(row)
    for task_id in row.tasks:
        backend.validate_custom_model(get_task_descriptor(task_id), ref)
    return result


async def _finish(model_id: uuid.UUID, **values: Any) -> bool:
    """Compare-and-swap out of `validating`. False means someone else moved the row on."""
    async with get_sessionmaker()() as s:
        res = await s.execute(
            sa.update(CustomModel)
            .where(CustomModel.id == model_id, CustomModel.status == "validating")
            .values(claimed_by=None, lease_expires_at=None, **values)
            .returning(CustomModel.id)
        )
        moved = res.first() is not None
        await s.commit()
    return moved


async def run_validate(model_id: uuid.UUID) -> None:
    async with get_sessionmaker()() as s:
        row = await s.get(CustomModel, model_id)
        if row is None:
            logger.warning("Custom model %s vanished before it was validated", model_id)
            return
        s.expunge(row)  # detached: the worker thread reads and updates it without a session

    try:
        result = await run_in_executor(validate_executor, validate_sync, row)
    except (CustomModelFileError, ConfigError) as e:
        logger.info("Custom model %s failed validation: %s", model_id, e)
        await _finish(model_id, status="failed", last_error=str(e)[:2000])
        _forget(row)
        return

    await _finish(
        model_id,
        status="ready",
        last_error=None,
        **{k: v for k, v in vars(result).items() if v is not None},
    )
    logger.info("Custom model %s is ready", model_id)


def _forget(row: CustomModel) -> None:
    """A model that failed validation must not keep a half-unpacked directory around."""
    try:
        shutil.rmtree(Path(custom_models.local_dir_for(row)).parent, ignore_errors=True)
    except Exception:
        logger.debug("Could not clean up files of failed model %s", row.id, exc_info=True)


async def handle_failure(model_id: uuid.UUID, exc: BaseException) -> None:
    """An unexpected error (typically the network): retry after a delay, or fail once attempts run out."""
    status = await queue.release_or_fail(queue.VALIDATE, model_id, f"{type(exc).__name__}: {exc}")
    logger.warning("Custom model %s validation errored (%s); now %s", model_id, exc, status)
