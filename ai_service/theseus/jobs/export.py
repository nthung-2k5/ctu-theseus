"""Run one export job: convert the model artifact if needed, then assemble the bundle.

Framework-neutral since trainer backends became a plugin system: the actual model conversion and
golden-sample prediction go through the run's own `TrainerBackend`, not Ludwig directly.

Replaces ai_service/tasks/export.py plus the gateway two-phase dance (converting -> event ->
assembling -> in-memory queue). One job now walks the whole status flow on a single row:

    pending -> converting -> assembling -> ready | failed

Conversion is skipped when the artifact already exists (an earlier export of another format built
from the same artifact already produced it). The gateway lazy-reconcile-on-GET recovery is unnecessary: there
is no dropped event to recover from, and a crashed job is re-queued by lease expiry / startup
recovery, which is safe because both conversion and assembly rewrite the same S3 keys.
"""

import logging
import os
import threading
import uuid
from typing import Any

import pandas as pd
import sqlalchemy as sa

from theseus import constants as C
from theseus.backends.base import TrainerBackend
from theseus.backends.registry import get_backend
from theseus.db.base import get_sessionmaker
from theseus.db.models import ModelExport, TrainingRun
from theseus.export import bundle
from theseus.export.registry import find_export_format
from theseus.jobs import queue
from theseus.jobs.executors import export_executor, run_in_executor
from theseus.services import storage
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.export")

# Loading a model puts it on the GPU. Assembly (zipping) runs two at a time, but model conversion
# is serialized: this is the old worker one-task-at-a-time behavior, kept where it matters.
_conversion_lock = threading.Lock()


def _plain(v: Any) -> Any:
    """A pandas/numpy scalar as a plain Python one. json.dumps has no numpy support and would
    otherwise silently stringify it through the default=str fallback instead of writing a number."""
    return v.item() if hasattr(v, "item") else v


def _shape_input_value(input_columns: list[str], sample_row: "pd.Series") -> tuple[str | None, Any]:
    """(inputColumn, inputValue) for expected.json.

    A single input column keeps the original scalar shape (an s3:// URI for file-backed
    modalities, or the inline text/scalar), which bundle assembly turns into sample/input.<ext>.
    A tabular model (more than one input column) becomes a {column: value} record written as
    sample/input.json; the devkit client predict() takes that same dict shape.
    """
    if len(input_columns) == 1:
        return input_columns[0], _plain(sample_row[input_columns[0]])
    return None, {c: _plain(sample_row[c]) for c in input_columns}


def _build_golden_sample(backend: type[TrainerBackend], model_dir: str, dataset_key: str) -> dict[str, Any] | None:
    """Run one real test-split row through the trained model.

    Returns the input/output pair the devkit verify script checks its own (re-implemented)
    preprocessing against. Returns None rather than raising for environmental problems, since a
    missing snapshot should skip the verification artifact rather than fail the export.
    """
    try:
        model = backend.load(model_dir)
        df = pd.read_parquet(storage.s3fs_path(C.BUCKET_DATASETS, dataset_key))
        rows = df[df[C.SPLIT_COLUMN_NAME] == "test"] if C.SPLIT_COLUMN_NAME in df.columns else df
        if len(rows) == 0:
            rows = df
        sample = rows.iloc[[0]]

        predictions = model.predict(sample)
        # threshold=0.0: this is a reference fixture, not a filtered result. The verify script
        # wants the full distribution to compare.
        predicted = model.golden_prediction(predictions, threshold=0.0)
        input_column, input_value = _shape_input_value(model.input_columns, sample.iloc[0])
        return {
            "schemaVersion": 1,
            "inputColumn": input_column,
            "inputValue": input_value,
            "outputColumn": model.output.name,
            "outputType": model.output.kind,
            "predictions": predicted,
        }
    except (AttributeError, TypeError, KeyError, IndexError, NameError):
        # A bug in this function, not a data problem: surface it. These are exactly the failures
        # that used to vanish into `return None`, silently disabling every bundle verify script.
        logger.exception("Bug while building the golden sample for export verification")
        raise
    except Exception:
        logger.exception("Failed to build golden sample for export verification, skipping expected.json")
        return None


def _convert(run_id: str, backend: type[TrainerBackend], artifact_id: str, dataset_key: str, export_id: str) -> None:
    """Sync: download the trained model, export it, upload the artifact and the golden sample."""
    with _conversion_lock:
        model_dir = storage.find_model_dir(storage.download_model(run_id))
        workdir = os.path.join(str(get_settings().temp_dir), "exports", export_id)
        os.makedirs(workdir, exist_ok=True)

        model = backend.load(model_dir)
        artifact = backend.artifacts[artifact_id]
        export_path = backend.convert(model, artifact_id, workdir)
        storage.upload_file(C.BUCKET_MODELS, storage.export_key(run_id, artifact.filename), export_path)

        # Best effort, and idempotent (same S3 key), so re-running per artifact is fine.
        golden = _build_golden_sample(backend, model_dir, dataset_key)
        if golden is not None:
            storage.upload_json(C.BUCKET_MODELS, storage.expected_sample_key(run_id), golden)
        storage.cleanup_temp("exports", export_id)


async def run_export(export_id: uuid.UUID) -> None:
    """Run a claimed export job. Raises on failure; the dispatcher then calls handle_failure."""
    async with get_sessionmaker()() as s:
        row = await s.get(ModelExport, export_id)
        run = await s.get(TrainingRun, row.run_id) if row else None
    if row is None or run is None:
        return
    run_id = str(row.run_id)
    export_format = find_export_format(row.format)
    if export_format is None:
        # The plugin class was removed between enqueue and run; retrying cannot help.
        raise ValueError(f"Export format {row.format!r} is no longer installed")
    backend = get_backend(run.backend)
    artifact = backend.artifacts[export_format.artifact]
    logger.info(
        "Starting export %s for run %s (format %s, backend %s, artifact %s)",
        export_id, run_id, row.format, backend.id, artifact.id,
    )  # fmt: skip

    if not await run_in_executor(
        None, storage.file_exists, C.BUCKET_MODELS, storage.export_key(run_id, artifact.filename)
    ):
        dataset_key = storage.snapshot_parquet_key(str(run.dataset_version_id))
        await run_in_executor(
            export_executor, _convert, run_id, backend, export_format.artifact, dataset_key, str(export_id)
        )

    # Guarded: only a job still in `converting` may advance. Zero rows means it was recovered or
    # failed by someone else in the meantime, so do nothing further.
    async with get_sessionmaker()() as s:
        moved = await s.execute(
            sa.update(ModelExport)
            .where(ModelExport.id == export_id, ModelExport.status == "converting")
            .values(status="assembling")
            .returning(ModelExport.id)
        )
        advanced = moved.first() is not None
        await s.commit()
    if not advanced:
        return

    # Owns the final ready / failed transition and never raises.
    await bundle.build_bundle(export_id)


async def handle_failure(job_id: uuid.UUID, exc: BaseException) -> None:
    """An attempt failed: retry after a delay, or fail for good once attempts are exhausted."""
    error = f"{type(exc).__name__}: {exc}"
    logger.error("Export %s failed: %s", job_id, error)
    await queue.release_or_fail(queue.EXPORT, job_id, error, final_values={"failed_message": error[:500]})
