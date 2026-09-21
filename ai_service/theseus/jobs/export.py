"""Run one export job: convert the model artifact if needed, then assemble the bundle.

Replaces ai_service/tasks/export.py plus the gateway two-phase dance (converting -> event ->
assembling -> in-memory queue). One job now walks the whole status flow on a single row:

    pending -> converting -> assembling -> ready | failed

Conversion is skipped when the artifact already exists (an earlier export of another tier or
language already produced it). The gateway lazy-reconcile-on-GET recovery is unnecessary: there
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
from ludwig.api import LudwigModel

from theseus import constants as C
from theseus.db.base import get_sessionmaker
from theseus.db.models import ModelExport, TrainingRun
from theseus.export import bundle
from theseus.jobs import queue
from theseus.jobs.executors import export_executor, run_in_executor
from theseus.services import storage
from theseus.services.predict import parse_prediction_row
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.export")

# Loading a Ludwig model puts it on the GPU. Assembly (zipping) runs two at a time, but model
# conversion is serialized: this is the old worker one-task-at-a-time behavior, kept where it matters.
_conversion_lock = threading.Lock()


def _plain(v: Any) -> Any:
    """A pandas/numpy scalar as a plain Python one. json.dumps has no numpy support and would
    otherwise silently stringify it through the default=str fallback instead of writing a number."""
    return v.item() if hasattr(v, "item") else v


def _shape_input_value(input_features: list[Any], sample_row: "pd.Series") -> tuple[str | None, Any]:
    """(inputColumn, inputValue) for expected.json.

    A single input feature keeps the original scalar shape (an s3:// URI for file-backed
    modalities, or the inline text/scalar), which bundle assembly turns into sample/input.<ext>.
    A tabular model (more than one input feature) becomes a {column: value} record written as
    sample/input.json; the devkit client predict() takes that same dict shape.
    """
    if len(input_features) == 1:
        return input_features[0].column, _plain(sample_row[input_features[0].column])
    return None, {f.column: _plain(sample_row[f.column]) for f in input_features}


def _build_golden_sample(ludwig_model_dir: str, dataset_key: str) -> dict[str, Any] | None:
    """Run one real test-split row through the trained model.

    Returns the input/output pair the devkit verify script checks its own (re-implemented)
    preprocessing against. Returns None rather than raising for environmental problems, since a
    missing snapshot should skip the verification artifact rather than fail the export.
    """
    try:
        model = LudwigModel.load(ludwig_model_dir)
        df = pd.read_parquet(storage.s3fs_path(C.BUCKET_DATASETS, dataset_key))
        rows = df[df[C.SPLIT_COLUMN_NAME] == "test"] if C.SPLIT_COLUMN_NAME in df.columns else df
        if len(rows) == 0:
            rows = df
        sample = rows.iloc[[0]]

        input_features = model.config_obj.input_features
        output_feature = model.config_obj.output_features[0]

        predictions, _ = model.predict(dataset=sample)
        assert isinstance(predictions, pd.DataFrame)

        idx2str = None
        if output_feature.type == "category" and model.training_set_metadata:
            idx2str = model.training_set_metadata.get(output_feature.name, {}).get("idx2str")

        # threshold=0.0: this is a reference fixture, not a filtered result. The verify script
        # wants the full distribution to compare.
        predicted = parse_prediction_row(output_feature.name, output_feature.type, predictions, idx2str, threshold=0.0)
        input_column, input_value = _shape_input_value(input_features, sample.iloc[0])
        return {
            "schemaVersion": 1,
            "inputColumn": input_column,
            "inputValue": input_value,
            "outputColumn": output_feature.name,
            "outputType": output_feature.type,
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


def _convert(run_id: str, export_format: str, dataset_key: str, export_id: str) -> None:
    """Sync: download the trained model, export it, upload the artifact and the golden sample."""
    with _conversion_lock:
        model_dir = storage.find_model_dir(storage.download_model(run_id))
        export_path = os.path.join(str(get_settings().temp_dir), "exports", export_id, f"model.{export_format}")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        model = LudwigModel.load(model_dir)
        if export_format == "torchscript":
            model.export_model(export_path, format="torch_export")
        elif export_format == "onnx":
            model.export_model(export_path, format="onnx")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        storage.upload_file(C.BUCKET_MODELS, storage.export_key(run_id, export_format), export_path)

        # Best effort, and idempotent (same S3 key), so re-running per format is fine.
        golden = _build_golden_sample(model_dir, dataset_key)
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
    run_id, fmt = str(row.run_id), row.format
    logger.info("Starting export %s for run %s (format %s, tier %s)", export_id, run_id, fmt, row.tier)

    if not await run_in_executor(None, storage.file_exists, C.BUCKET_MODELS, storage.export_key(run_id, fmt)):
        dataset_key = storage.snapshot_parquet_key(str(run.dataset_version_id))
        await run_in_executor(export_executor, _convert, run_id, fmt, dataset_key, str(export_id))

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
