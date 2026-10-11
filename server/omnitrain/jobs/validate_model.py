"""Validate a bring-your-own model: pin it, fetch it, check it, then mark it `ready`.

A custom model row is created `uploaded` (a Hub reference, or an upload the browser has finished) and
claimed off the `custom_models` table like any other job (see jobs/queue.py). The job:

  1. Hub: look the repo up, refuse a private/gated or oversized one, and PIN the commit sha as the
     model's revision. Upload: download the bundle once, hash it and unpack it (`import_upload`).
  2. Generic safety checks on the files (`custom_models.check_tree`): no pickles or code, no symlinks,
     within the size limit.
  3. The backend's own check (`TrainerBackend.validate_custom_model`) for every task the model is offered
     for (all the tasks its kind serves): is this really a causal LM, does it have a config.json, and so on.

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

from omnitrain.backends.base import ConfigError, CustomModelKind
from omnitrain.backends.registry import get_backend
from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import CustomModel
from omnitrain.jobs import queue
from omnitrain.jobs.executors import run_in_executor, validate_executor
from omnitrain.services import custom_models, integrations
from omnitrain.services.custom_models import CustomModelFileError
from omnitrain.services.model_catalog import kind_and_tasks
from omnitrain.services.task_registry import get_task_descriptor
from omnitrain.settings import get_settings

logger = logging.getLogger("omnitrain.jobs.validate_model")


@dataclass
class Validated:
    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    # A Hub model that needs its owner's token. Lives in the row's `spec`, not a column.
    private: bool | None = None


def _kind_for(row: CustomModel) -> tuple[CustomModelKind, list[str]]:
    kind, tasks = kind_and_tasks(get_backend(row.backend), row.kind)
    if kind is None:
        raise ConfigError(f"'{row.backend}' does not offer a '{row.kind}' model")
    return kind, tasks


def validate_sync(row: CustomModel, token: str | None = None) -> Validated:
    """Everything that touches the network or the disk. Runs in a worker thread; mutates the detached
    `row` only to carry the pin forward, and returns what to persist. `token` is the owner's Hugging Face
    token, for a private or gated Hub model."""
    limit = get_settings().max_custom_model_bytes
    backend = get_backend(row.backend)
    kind, tasks = _kind_for(row)
    if kind.unavailable_reason:
        raise ConfigError(kind.unavailable_reason)

    result = Validated()
    if not kind.materialize:
        pass  # the framework fetches its own weights by name; nothing to pin or check on disk
    elif row.source_kind == "hub":
        pin = custom_models.hub_pin(row.source_ref, token)
        if pin.size_bytes > limit:
            raise CustomModelFileError(
                f"The model is {pin.size_bytes // 2**20} MiB, over the {limit // 2**20} MiB limit"
            )
        row.revision = result.revision = pin.revision
        result.private = pin.private
        path = custom_models.ensure_local(row, token)
        result.size_bytes = custom_models.check_tree(path, max_bytes=limit)
    else:
        imported = custom_models.import_upload(row)
        row.sha256 = result.sha256 = imported.sha256
        result.size_bytes = custom_models.check_tree(imported.path, max_bytes=limit)

    ref = custom_models.ref_for(row)
    for task_id in tasks:
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

    # Only the model's own owner's token is ever used: a global model has no owner, so it never carries one.
    token = await integrations.huggingface_token_for(row.owner_user_id) if row.source_kind == "hub" else None
    try:
        result = await run_in_executor(validate_executor, validate_sync, row, token)
    except (CustomModelFileError, ConfigError) as e:
        logger.info("Custom model %s failed validation: %s", model_id, e)
        message = str(e)
        if row.source_kind == "hub" and row.owner_user_id is None and token is None:
            message += " (Models shared with everyone can only come from public repositories.)"
        await _finish(model_id, status="failed", last_error=message[:2000])
        _forget(row)
        return

    values: dict[str, Any] = {k: v for k, v in vars(result).items() if v is not None and k != "private"}
    if result.private is not None:
        values["spec"] = {**(row.spec or {}), "private": result.private}
    await _finish(model_id, status="ready", last_error=None, **values)
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
