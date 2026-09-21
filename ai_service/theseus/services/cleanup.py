"""S3 teardown for project / version / run deletes.

Ported from server/lib/cleanup.ts. Cleanup is best-effort by design: a failed object delete is
logged and skipped, never allowed to block deleting the database row the user asked for.

Callers delete the S3 objects BEFORE the database rows (the object keys live in those rows).
"""

import asyncio
import logging
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.db.models import DatasetItem, DatasetVersion, InferenceJob, TrainingRun
from theseus.services import storage

logger = logging.getLogger(__name__)


def _best_effort(fn: Callable[..., Any], *args: Any) -> None:
    try:
        fn(*args)
    except Exception:
        logger.warning("S3 cleanup step failed: %s%s", fn.__name__, args, exc_info=True)


async def _run(steps: Iterable[tuple[Callable[..., Any], tuple[Any, ...]]]) -> None:
    loop = asyncio.get_running_loop()
    await asyncio.gather(*(loop.run_in_executor(None, _best_effort, fn, *args) for fn, args in steps))


def _version_steps(version_id: uuid.UUID | str, version_tag: str | None) -> list[tuple[Callable[..., Any], tuple]]:
    # The draft (no version tag) has no parquet or manifest; only real snapshots do.
    if version_tag is None:
        return []
    v = str(version_id)
    return [
        (storage.delete_file, (C.BUCKET_DATASETS, storage.snapshot_parquet_key(v))),
        (storage.delete_file, (C.BUCKET_DATASETS, storage.snapshot_manifest_key(v))),
    ]


def _run_steps(run_id: uuid.UUID | str, upload_keys: Iterable[str] = ()) -> list[tuple[Callable[..., Any], tuple]]:
    r = str(run_id)
    steps: list[tuple[Callable[..., Any], tuple]] = [
        (storage.delete_file, (C.BUCKET_TRAINING, storage.training_config_key(r))),
        (storage.delete_prefix, (C.BUCKET_TRAINING, storage.training_results_prefix(r))),
        (storage.delete_file, (C.BUCKET_TRAINING, storage.training_logs_key(r))),
        (storage.delete_prefix, (C.BUCKET_TRAINING, storage.evaluation_prefix(r))),
        # Covers model.{format}, expected.json, bundles/*.zip and batch predictions in one sweep.
        (storage.delete_prefix, (C.BUCKET_MODELS, f"{r}/")),
    ]
    keys = list(upload_keys)
    if keys:
        steps.append((storage.delete_files, (C.BUCKET_UPLOADS, keys)))
    return steps


async def cleanup_version_storage(version_id: uuid.UUID | str, version_tag: str | None) -> None:
    """Delete one snapshot version S3 objects. A no-op for the draft."""
    await _run(_version_steps(version_id, version_tag))


async def cleanup_run_storage(session: AsyncSession, run_id: uuid.UUID | str) -> None:
    """Delete one training run S3 objects: config, results, logs, exports, and pending inference uploads."""
    upload_keys = (
        (
            await session.execute(
                sa.select(InferenceJob.upload_key).where(
                    InferenceJob.run_id == run_id, InferenceJob.upload_key.is_not(None)
                )
            )
        )
        .scalars()
        .all()
    )
    await _run(_run_steps(run_id, upload_keys))


async def cleanup_project_storage(session: AsyncSession, project_id: uuid.UUID) -> None:
    """Delete every S3 object belonging to a project: pool files, snapshots, training artifacts, exports."""
    storage_urls = (
        (
            await session.execute(
                sa.select(DatasetItem.storage_url).where(
                    DatasetItem.dataset_id == project_id, DatasetItem.storage_url.is_not(None)
                )
            )
        )
        .scalars()
        .all()
    )
    versions = (
        await session.execute(
            sa.select(DatasetVersion.id, DatasetVersion.version_tag).where(DatasetVersion.dataset_id == project_id)
        )
    ).all()
    runs = (
        (await session.execute(sa.select(TrainingRun.id).where(TrainingRun.project_id == project_id))).scalars().all()
    )
    upload_keys = (
        (
            await session.execute(
                sa.select(InferenceJob.upload_key).where(
                    InferenceJob.run_id.in_(runs), InferenceJob.upload_key.is_not(None)
                )
            )
        )
        .scalars()
        .all()
        if runs
        else []
    )

    steps: list[tuple[Callable[..., Any], tuple]] = []
    if storage_urls:
        steps.append((storage.delete_files, (C.BUCKET_DATASETS, list(storage_urls))))
    for version_id, tag in versions:
        steps.extend(_version_steps(version_id, tag))
    for run_id in runs:
        steps.extend(_run_steps(run_id))
    if upload_keys:
        steps.append((storage.delete_files, (C.BUCKET_UPLOADS, list(upload_keys))))
    await _run(steps)
