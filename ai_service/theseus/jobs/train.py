"""Run one training job: dispatch to the run's trainer backend, live progress, cancel, evaluation
report, log upload.

Framework-neutral since trainer backends became a plugin system: everything here is common to any
backend (Ludwig or otherwise) — the run lifecycle, abort registration, log attach/detach/upload,
and the evaluation report upload. What changed relative to the pre-plugin worker still applies:

  * Progress goes straight to the event writer (a thread-safe queue put) instead of a
    run_coroutine_threadsafe(...).result(timeout=5) round trip per publish.
  * Abort is a threading.Event check (see abort.py), not a JetStream KV read with a 5 s timeout.
  * There is no redelivery, so no on_permanent_failure callback and no "do not publish failed on
    every retry" dance: the run either finishes here or fails here, exactly once.
  * Log streaming uses the contextvar-scoped handler, not a handler on the root logger.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from opentelemetry import trace
from sqlalchemy.dialects.postgresql import insert as pg_insert

from theseus import constants as C
from theseus.backends.base import LoadedModel, TrainContext
from theseus.backends.registry import get_backend
from theseus.db.base import get_sessionmaker
from theseus.db.models import RunEvaluation, TrainingRun
from theseus.events import get_event_writer, get_log_handler
from theseus.events.log_handler import current_run_id
from theseus.jobs import abort
from theseus.jobs.executors import run_in_executor, train_executor
from theseus.services import storage
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.train")
tracer = trace.get_tracer("theseus")


@dataclass
class RunSpec:
    id: uuid.UUID
    backend: str
    config: dict[str, Any]
    dataset_version_id: uuid.UUID
    config_key: str | None
    cancel_requested: bool


async def _load_run(run_id: uuid.UUID) -> RunSpec | None:
    async with get_sessionmaker()() as s:
        run = await s.get(TrainingRun, run_id)
        if run is None:
            return None
        return RunSpec(
            run.id, run.backend, run.config or {}, run.dataset_version_id, run.config_key,
            run.cancel_requested_at is not None,
        )  # fmt: skip


async def _save_evaluation(run_id: uuid.UUID, **values: Any) -> None:
    """Upsert the run's single evaluation row (run_id is its primary key)."""
    values = {**values, "evaluated_at": datetime.now(UTC)}
    stmt = pg_insert(RunEvaluation).values(run_id=run_id, **values)
    async with get_sessionmaker()() as s:
        await s.execute(stmt.on_conflict_do_update(index_elements=["run_id"], set_=values))
        await s.commit()


async def _evaluate(backend, model: LoadedModel, run_id: uuid.UUID, dataset_path: str) -> None:
    """Best-effort evaluation report. A bug here must never fail a run that trained fine."""
    rid = str(run_id)
    try:
        df = await run_in_executor(train_executor, pd.read_parquet, dataset_path)
        result = await run_in_executor(
            train_executor, backend.evaluate, model, df, C.SPLIT_COLUMN_NAME, C.ITEM_ID_COLUMN_NAME
        )
        if result is None:
            logger.info("No data to evaluate for run %s, skipping evaluation report", rid)
            return
        report_key = storage.evaluation_report_key(rid)
        predictions_key = storage.evaluation_predictions_key(rid)
        await run_in_executor(train_executor, storage.upload_json, C.BUCKET_TRAINING, report_key, result.report)
        await run_in_executor(
            train_executor, result.predictions.to_parquet, storage.s3fs_path(C.BUCKET_TRAINING, predictions_key)
        )
        overall = result.report.get("overall") or {}
        await _save_evaluation(
            run_id,
            status="success",
            split=result.report["split"],
            report_key=report_key,
            predictions_key=predictions_key,
            report=result.report,
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

        backend = get_backend(run.backend)

        # Ludwig increments results_run_N instead of overwriting. Start from a clean output prefix
        # so there is exactly one candidate for export and inference to serve, whatever backend
        # wrote it.
        output_prefix = storage.training_results_prefix(rid)
        await run_in_executor(None, storage.delete_prefix, C.BUCKET_TRAINING, output_prefix)

        dataset_path = storage.s3fs_path(C.BUCKET_DATASETS, storage.snapshot_parquet_key(str(run.dataset_version_id)))
        output_path = storage.s3fs_path(C.BUCKET_TRAINING, output_prefix)
        workdir = str(get_settings().temp_dir / "train" / rid)

        ctx = TrainContext(
            run_id=rid,
            config=run.config,
            dataset_uri=dataset_path,
            output_uri=output_path,
            workdir=workdir,
            _heartbeat=lambda: writer.touch(rid),
            _check_abort=lambda: _check_abort(rid, abort_event),
            _report=lambda epoch, split, metrics: writer.metric(rid, epoch, split, metrics),
        )

        with tracer.start_as_current_span(f"{backend.id}.train"):
            model = await run_in_executor(train_executor, backend.train, ctx)

        if abort_event.is_set():
            logger.info("Training aborted for run %s", rid)
            writer.status(rid, "canceled")
            return

        await _evaluate(backend, model, run_id, dataset_path)
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


def _check_abort(run_id: str, abort_event) -> None:
    if abort_event.is_set():
        raise abort.TrainingAborted(f"Training aborted for run {run_id}")


async def handle_failure(job_id: uuid.UUID, exc: BaseException) -> None:
    """Backstop for a bug that escaped run_train (which normally never raises). Training is never retried."""
    logger.error("Training job %s escaped its handler: %s", job_id, exc)
    get_event_writer().status(job_id, "failed", f"{type(exc).__name__}: {exc}")
