"""Run one training job: Ludwig training, live progress, cancel, evaluation report, log upload.

Replaces ai_service/tasks/train.py. What changed relative to the NATS worker:

  * Progress goes straight to the event writer (a thread-safe queue put) instead of a
    run_coroutine_threadsafe(...).result(timeout=5) round trip per publish.
  * Abort is a threading.Event check (see abort.py), not a JetStream KV read with a 5 s timeout.
  * There is no redelivery, so no on_permanent_failure callback and no "do not publish failed on
    every retry" dance: the run either finishes here or fails here, exactly once.
  * Log streaming uses the contextvar-scoped handler, not a handler on the root logger.
"""

import functools
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import ludwig.constants as ludwig_consts
import pandas as pd
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from opentelemetry import trace
from sqlalchemy.dialects.postgresql import insert as pg_insert

from theseus import constants as C
from theseus.db.base import get_sessionmaker
from theseus.db.models import RunEvaluation, TrainingRun
from theseus.events import get_event_writer, get_log_handler
from theseus.events.log_handler import current_run_id
from theseus.jobs import abort
from theseus.jobs.executors import run_in_executor, train_executor
from theseus.services import storage
from theseus.services.evaluate import build_evaluation_report
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.train")
tracer = trace.get_tracer("theseus")


@dataclass
class RunSpec:
    id: uuid.UUID
    dataset_version_id: uuid.UUID
    config_key: str | None
    cancel_requested: bool


def _extract_metrics(feature_metrics: dict[str, dict[str, list[Any]]]) -> dict[str, float]:
    """Flatten every metric Ludwig tracks for a split into {metricName: latest value}.

    Per-output-feature metrics are namespaced `{featureName}.{metric}` so a multi-output config
    cannot collide; the `combined` feature (Ludwig aggregate across outputs) stays unprefixed.
    """
    out: dict[str, float] = {}
    for feature_name, per_metric in feature_metrics.items():
        prefix = "" if feature_name == ludwig_consts.COMBINED else f"{feature_name}."
        for metric_name, history in per_metric.items():
            if history:
                out[f"{prefix}{metric_name}"] = float(history[-1].value)
    return out


class TrainingProgressCallback(Callback):
    """Ludwig callback (runs in the training thread): heartbeat, abort check, per-epoch metrics."""

    def __init__(self, run_id: str, abort_event) -> None:
        self.run_id = run_id
        self.abort_event = abort_event

    def _check_abort(self) -> None:
        if self.abort_event.is_set():
            raise abort.TrainingAborted(f"Training aborted for run {self.run_id}")

    def on_epoch_start(self, trainer, progress_tracker, save_path, **kwargs):
        # Epoch-end events can be minutes apart on large datasets; this keeps the run looking alive.
        self._check_abort()
        get_event_writer().touch(self.run_id)

    def on_epoch_end(self, trainer, progress_tracker, save_path, **kwargs):
        self._check_abort()
        epoch = progress_tracker.epoch
        trace.get_current_span().add_event(f"epoch {epoch} end", {"epoch": epoch})
        splits = {
            "train": progress_tracker.train_metrics,
            "validation": progress_tracker.validation_metrics,
            "test": progress_tracker.test_metrics,
        }
        writer = get_event_writer()
        for split_name, split_metrics in splits.items():
            metrics = _extract_metrics(split_metrics)
            if metrics:
                writer.metric(self.run_id, epoch, split_name, metrics)


async def _load_run(run_id: uuid.UUID) -> RunSpec | None:
    async with get_sessionmaker()() as s:
        run = await s.get(TrainingRun, run_id)
        if run is None:
            return None
        return RunSpec(run.id, run.dataset_version_id, run.config_key, run.cancel_requested_at is not None)


async def _save_evaluation(run_id: uuid.UUID, **values: Any) -> None:
    """Upsert the run's single evaluation row (run_id is its primary key)."""
    values = {**values, "evaluated_at": datetime.now(UTC)}
    stmt = pg_insert(RunEvaluation).values(run_id=run_id, **values)
    async with get_sessionmaker()() as s:
        await s.execute(stmt.on_conflict_do_update(index_elements=["run_id"], set_=values))
        await s.commit()


async def _evaluate(model: LudwigModel, run_id: uuid.UUID, dataset_path: str) -> None:
    """Best-effort evaluation report. A bug here must never fail a run that trained fine."""
    rid = str(run_id)
    try:
        df = await run_in_executor(train_executor, pd.read_parquet, dataset_path)
        result = await run_in_executor(
            train_executor, build_evaluation_report, model, df, C.SPLIT_COLUMN_NAME, C.ITEM_ID_COLUMN_NAME
        )
        if result is None:
            logger.info("No data to evaluate for run %s, skipping evaluation report", rid)
            return
        report, predictions_df = result
        report_key = storage.evaluation_report_key(rid)
        predictions_key = storage.evaluation_predictions_key(rid)
        await run_in_executor(train_executor, storage.upload_json, C.BUCKET_TRAINING, report_key, report)
        await run_in_executor(
            train_executor, predictions_df.to_parquet, storage.s3fs_path(C.BUCKET_TRAINING, predictions_key)
        )
        overall = report.get("overall") or {}
        await _save_evaluation(
            run_id,
            status="success",
            split=report["split"],
            report_key=report_key,
            predictions_key=predictions_key,
            report=report,
            accuracy=overall.get("accuracy"),
            macro_f1=overall.get("macroF1"),
            failed_message=None,
        )
    except Exception as e:
        logger.exception("Evaluation report failed for run %s (the training result is unaffected)", rid)
        try:
            await _save_evaluation(run_id, status="failed", failed_message=str(e)[:2000])
        except Exception:
            logger.exception("Could not record the evaluation failure for run %s", rid)


async def run_train(run_id: uuid.UUID) -> None:
    """Run a claimed training job to a terminal status. Never raises."""
    writer = get_event_writer()
    handler = get_log_handler()
    rid = str(run_id)
    loop_token = current_run_id.set(rid)
    # Register the abort Event BEFORE reading the DB cancel flag (see abort.py for why the order matters).
    abort_event = abort.register(rid)
    log_path = get_settings().temp_dir / "logs" / f"{rid}.log"
    handler.attach(rid, log_path)
    try:
        run = await _load_run(run_id)
        if run is None:
            logger.error("Training run %s vanished before it started", rid)
            return
        if run.cancel_requested:
            abort_event.set()
        if abort_event.is_set():
            logger.info("Run %s was canceled before training started", rid)
            writer.status(rid, "canceled")
            return

        writer.status(rid, "running")
        logger.info("Starting training for run %s", rid)

        if not run.config_key or not await run_in_executor(
            None, storage.file_exists, C.BUCKET_TRAINING, run.config_key
        ):
            raise ValueError(f"No training config found at {run.config_key} for run {rid}")

        # Ludwig increments results_run_N instead of overwriting. Start from a clean output prefix
        # so there is exactly one candidate for export and inference to serve.
        output_prefix = storage.training_results_prefix(rid)
        await run_in_executor(None, storage.delete_prefix, C.BUCKET_TRAINING, output_prefix)

        config_path = storage.s3fs_path(C.BUCKET_TRAINING, run.config_key)
        dataset_path = storage.s3fs_path(C.BUCKET_DATASETS, storage.snapshot_parquet_key(str(run.dataset_version_id)))
        output_path = storage.s3fs_path(C.BUCKET_TRAINING, output_prefix)

        model = LudwigModel(
            config=config_path,
            logging_level=logging.INFO,
            callbacks=[TrainingProgressCallback(rid, abort_event)],
        )
        with tracer.start_as_current_span("ludwig.train"):
            await run_in_executor(
                train_executor,
                functools.partial(
                    model.train, dataset=dataset_path, output_directory=output_path, experiment_name="results"
                ),
            )

        if abort_event.is_set():
            logger.info("Training aborted for run %s", rid)
            writer.status(rid, "canceled")
            return

        await _evaluate(model, run_id, dataset_path)
        writer.status(rid, "succeeded")
        logger.info("Training completed for run %s", rid)

    except abort.TrainingAborted:
        logger.info("Training aborted for run %s", rid)
        writer.status(rid, "canceled")
    except Exception as e:
        logger.exception("Training failed for run %s", rid)
        writer.status(rid, "failed", f"{type(e).__name__}: {e}")
    finally:
        abort.unregister(rid)
        handler.detach(rid)
        current_run_id.reset(loop_token)
        if log_path.exists():
            try:
                await run_in_executor(
                    None, storage.upload_file, C.BUCKET_TRAINING, storage.training_logs_key(rid), str(log_path)
                )
            except Exception:
                logger.warning("Failed to upload the training log for run %s", rid, exc_info=True)
            log_path.unlink(missing_ok=True)


async def handle_failure(job_id: uuid.UUID, exc: BaseException) -> None:
    """Backstop for a bug that escaped run_train (which normally never raises). Training is never retried."""
    logger.error("Training job %s escaped its handler: %s", job_id, exc)
    get_event_writer().status(job_id, "failed", f"{type(exc).__name__}: {exc}")
