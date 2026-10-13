"""The part of a training job that does not care where it runs.

`execute` is everything between "the run is running" and its terminal status: check the compiled config exists,
clear the results prefix, call the trainer backend, then build the evaluation report. It is synchronous (the
caller picks the thread) and imports no database or event-writer code. All it needs from its surroundings is S3 and
a `ProgressSink`, which is what lets the same function run in the API process's `train` lane (`jobs/train.py`,
whose sink writes to the event writer) and on a remote GPU worker (whose sink would post to the API).

That is also the rule for any backend's `train()` and `evaluate()`: touch S3 and the `TrainContext` callbacks only.
Anything that reads Postgres there could not run on a worker.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
from opentelemetry import trace

from omnitrain.backends.base import LoadedModel, TrainContext, TrainerBackend
from omnitrain.backends.registry import get_backend
from omnitrain.core import constants as C
from omnitrain.services import storage
from omnitrain.settings import get_settings

logger = logging.getLogger("omnitrain.jobs.train")
tracer = trace.get_tracer("omnitrain")


@dataclass
class RunSpec:
    id: uuid.UUID
    backend: str
    config: dict[str, Any]
    dataset_version_id: uuid.UUID
    config_key: str | None


class ProgressSink(Protocol):
    """Where a running backend's progress goes. Called from the training thread."""

    def heartbeat(self) -> None: ...

    def check_abort(self) -> None:
        """Raise `omnitrain.jobs.abort.TrainingAborted` if the run was asked to stop."""

    def report(self, epoch: int, split: str, metrics: dict[str, float]) -> None: ...


def evaluate(
    backend: type[TrainerBackend], model: LoadedModel, run_id: uuid.UUID, dataset_path: str
) -> dict[str, Any] | None:
    """Best-effort evaluation report. A bug here must never fail a run that trained fine.

    Returns the `run_evaluations` columns to store (a `success` row, or a `failed` one carrying the error), or None
    when there is nothing to evaluate. Only the report and predictions are uploaded here; recording the row is the
    caller's job, since that is the part that needs the database.
    """
    rid = str(run_id)
    try:
        df = pd.read_parquet(dataset_path)
        result = backend.evaluate(model, df, C.SPLIT_COLUMN_NAME, C.ITEM_ID_COLUMN_NAME)
        if result is None:
            logger.info("No data to evaluate for run %s, skipping evaluation report", rid)
            return None
        report_key = storage.evaluation_report_key(rid)
        predictions_key = storage.evaluation_predictions_key(rid)
        storage.upload_json(C.BUCKET_TRAINING, report_key, result.report)
        result.predictions.to_parquet(storage.s3fs_path(C.BUCKET_TRAINING, predictions_key))
        overall = result.report.get("overall") or {}
        return {
            "status": "success",
            "split": result.report["split"],
            "report_key": report_key,
            "predictions_key": predictions_key,
            "report": result.report,
            "accuracy": overall.get("accuracy"),
            "macro_f1": overall.get("macroF1"),
            "failed_message": None,
        }
    except Exception as e:
        logger.exception("Evaluation report failed for run %s (the training result is unaffected)", rid)
        return {"status": "failed", "failed_message": str(e)[:2000]}


def execute(spec: RunSpec, sink: ProgressSink) -> dict[str, Any] | None:
    """Train and evaluate one run. Returns the evaluation columns to record (see `evaluate`), or None.

    Raises on a training failure, and `TrainingAborted` (from `sink.check_abort`) when the run was canceled.
    """
    rid = str(spec.id)
    if not spec.config_key or not storage.file_exists(C.BUCKET_TRAINING, spec.config_key):
        raise ValueError(f"No training config found at {spec.config_key} for run {rid}")

    backend = get_backend(spec.backend)

    # Ludwig increments results_run_N instead of overwriting. Start from a clean output prefix
    # so there is exactly one candidate for export and inference to serve, whatever backend
    # wrote it.
    output_prefix = storage.training_results_prefix(rid)
    storage.delete_prefix(C.BUCKET_TRAINING, output_prefix)

    dataset_path = storage.s3fs_path(C.BUCKET_DATASETS, storage.snapshot_parquet_key(str(spec.dataset_version_id)))
    ctx = TrainContext(
        run_id=rid,
        config=spec.config,
        dataset_uri=dataset_path,
        output_uri=storage.s3fs_path(C.BUCKET_TRAINING, output_prefix),
        workdir=str(get_settings().temp_dir / "train" / rid),
        _heartbeat=sink.heartbeat,
        _check_abort=sink.check_abort,
        _report=sink.report,
    )

    with tracer.start_as_current_span(f"{backend.id}.train"):
        model = backend.train(ctx)

    # A cancel that landed after the last epoch boundary: the backend finished, but the run is still canceled.
    sink.check_abort()

    return evaluate(backend, model, spec.id, dataset_path)
