import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Literal
from uuid import UUID

import ludwig.constants as ludwig_consts
import pandas as pd
from config import TEMP_DIR
from constants import ITEM_ID_COLUMN_NAME, SPLIT_COLUMN_NAME
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker
from opentelemetry import trace
from schema.command import Command
from schema.train_task import TrainTask
from services.evaluate import build_evaluation_report
from services.nats import nats_service
from services.storage import (
    BUCKET_DATASETS,
    BUCKET_TRAINING,
    delete_prefix,
    evaluation_predictions_key,
    evaluation_report_key,
    file_exists,
    s3fs_readable_path,
    training_logs_key,
    upload_file,
    upload_json,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("theseus-worker")

# ──────────────────────────────────────────────────────────────────
# Abort tracking
#
# Abort intent lives in NATS (THESEUS_ABORT_FLAGS, via
# `nats_service.is_aborted`/`set_abort_flag`), not in a process-local set —
# the gateway durably records the flag when it publishes the abort command
# (see server/lib/nats.ts `publishAbortCommand`), so a worker that's mid-
# restart when a cancel is requested still sees it once it comes back up,
# and a redelivered `TrainTask` for an already-aborted run can be recognized
# before training even starts (see `handle_train` below).
# ──────────────────────────────────────────────────────────────────


async def handle_command(data: Command) -> None:
    """Handle a command message (stop/abort). The gateway already wrote the
    durable abort flag before publishing this — this is just the immediate
    wake-up nudge, logged for visibility."""
    logger.info(f"Received abort command for run {data.run_id}")


# ──────────────────────────────────────────────────────────────────
# Log streaming — bridges Python `logging` records emitted during
# training onto NATS `log` RunEvents (throttled to ~10/s so a chatty
# Ludwig run can't flood the stream) and tees them to a local file that
# gets uploaded to S3 once the run finishes.
# ──────────────────────────────────────────────────────────────────
class RunLogHandler(logging.Handler):
    _LEVEL_MAP: ClassVar[dict[int, Literal["info", "warn", "error"]]] = {
        logging.DEBUG: "info",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(
        self,
        run_id: UUID,
        loop: asyncio.AbstractEventLoop,
        local_log_path: Path,
        rate_per_sec: float = 10.0,
    ):
        super().__init__()
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        self.run_id = run_id
        self.loop = loop
        self.interval = 1.0 / rate_per_sec
        self.queue: asyncio.Queue[tuple[Literal["info", "warn", "error"], str]] = (
            asyncio.Queue(maxsize=1000)
        )
        self._file = open(local_log_path, "a", encoding="utf-8")
        self._task = loop.create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            level, line = await self.queue.get()
            try:
                await nats_service.publish_log(str(self.run_id), line, level)
            except Exception:
                logger.warning(
                    f"Failed to publish log line for {self.run_id}", exc_info=True
                )
            await asyncio.sleep(self.interval)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        self._file.write(line + "\n")
        self._file.flush()
        level = self._LEVEL_MAP.get(record.levelno, "info")
        self.loop.call_soon_threadsafe(self._enqueue, level, line)

    def _enqueue(self, level: Literal["info", "warn", "error"], line: str) -> None:
        try:
            self.queue.put_nowait((level, line))
        except asyncio.QueueFull:
            pass

    async def aclose(self) -> None:
        """Not named `close` — that's `logging.Handler`'s synchronous method,
        and shadowing it with a coroutine would silently no-op whenever
        anything (e.g. `logging.shutdown()`) calls it without awaiting."""
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._file.close()
        super().close()


def _extract_metrics(
    feature_metrics: dict[str, dict[str, list[TrainerMetric]]],
) -> dict[str, float]:
    """Flatten every metric Ludwig tracks for a split — not just loss/accuracy
    on the combined feature — into `{metricName: value}` for the most recent
    step. Per-output-feature metrics are namespaced `{featureName}.{metric}`
    so a multi-output config can't collide; the `combined` feature (Ludwig's
    aggregate across all outputs) is left unprefixed.
    """
    metrics: dict[str, float] = {}
    for feature_name, per_metric in feature_metrics.items():
        prefix = "" if feature_name == ludwig_consts.COMBINED else f"{feature_name}."
        for metric_name, history in per_metric.items():
            if history:
                metrics[f"{prefix}{metric_name}"] = float(history[-1].value)
    return metrics


# ──────────────────────────────────────────────────────────────────
# Ludwig Callback for real-time progress reporting via NATS
# ──────────────────────────────────────────────────────────────────
class TrainingProgressCallback(Callback):
    """Ludwig callback that reports per-epoch metrics and heartbeats via NATS."""

    def __init__(self, run_id: UUID, loop: asyncio.AbstractEventLoop):
        self.run_id = run_id
        self.loop = loop

    def on_epoch_start(
        self, trainer, progress_tracker: ProgressTracker, save_path, **kwargs
    ):
        """Heartbeat so the gateway can distinguish "still training" from a
        wedged/dead worker between epoch-end events, which can be minutes
        apart on large datasets."""
        future = asyncio.run_coroutine_threadsafe(
            nats_service.publish_status(str(self.run_id), "running"), self.loop
        )
        try:
            future.result(timeout=5)
        except Exception as e:
            logger.warning(f"Failed to publish heartbeat: {e}")

    def on_epoch_end(
        self, trainer, progress_tracker: ProgressTracker, save_path, **kwargs
    ):
        """Called by Ludwig at the end of each training epoch."""
        # Check for abort. This is only checked at epoch boundaries — Ludwig
        # doesn't expose a batch-level callback hook here to check more
        # often, so a single very long epoch can't be interrupted mid-epoch.
        abort_future = asyncio.run_coroutine_threadsafe(
            nats_service.is_aborted(str(self.run_id)), self.loop
        )
        try:
            if abort_future.result(timeout=5):
                raise KeyboardInterrupt("Training aborted by user")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"Failed to check abort flag for {self.run_id}: {e}")

        epoch = progress_tracker.epoch
        trace.get_current_span().add_event(f"epoch {epoch} end", {"epoch": epoch})

        splits: dict[str, dict[str, dict[str, list[TrainerMetric]]]] = {
            "train": progress_tracker.train_metrics,
            "validation": progress_tracker.validation_metrics,
            "test": progress_tracker.test_metrics,
        }

        async def publish_all():
            for split_name, split_metrics in splits.items():
                metrics = _extract_metrics(split_metrics)
                if metrics:
                    await nats_service.publish_metric(
                        str(self.run_id), epoch, split_name, metrics
                    )

        # Publish progress to NATS (from sync callback via event loop)
        future = asyncio.run_coroutine_threadsafe(publish_all(), self.loop)

        # Wait for publish to complete (with timeout to not block training)
        try:
            future.result(timeout=5)
        except Exception as e:
            logger.warning(f"Failed to publish progress: {e}")


# ──────────────────────────────────────────────────────────────────
# Task Handlers
# ──────────────────────────────────────────────────────────────────
async def handle_train(data: TrainTask) -> None:
    """Handle a training task message."""
    run_id = data.run_id
    loop = asyncio.get_event_loop()

    logger.info(f"Starting training for run {run_id}")

    # A redelivered TrainTask (worker crash/restart) for a run that was
    # already aborted while the worker was down shouldn't start training at
    # all — the durable flag (unlike the old in-memory set) is still there.
    if await nats_service.is_aborted(str(run_id)):
        logger.info(f"Run {run_id} was aborted before training started — skipping")
        await nats_service.publish_status(str(run_id), "canceled")
        return

    local_log_path = TEMP_DIR / "logs" / f"{run_id}.log"
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handler = RunLogHandler(run_id, loop, local_log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    # Notify that training has started
    await nats_service.publish_status(str(run_id), "running")

    try:
        if not await asyncio.to_thread(file_exists, BUCKET_TRAINING, data.config_key):
            raise ValueError(
                f"No training config found at {data.config_key} for run {run_id}"
            )

        # Ludwig auto-increments `results_run_N` instead of overwriting, so a
        # re-dispatched or redelivered run would leave earlier attempts sitting
        # next to the new one — and whichever `find_model_dir` picked would be
        # the model that export and inference then serve. Start from a clean
        # output prefix so there is exactly one candidate.
        await asyncio.to_thread(delete_prefix, BUCKET_TRAINING, data.output_prefix)

        config_path = s3fs_readable_path(BUCKET_TRAINING, data.config_key)
        dataset_path = s3fs_readable_path(BUCKET_DATASETS, data.dataset_key)
        output_path = s3fs_readable_path(BUCKET_TRAINING, data.output_prefix)

        progress_callback = TrainingProgressCallback(run_id=run_id, loop=loop)

        model = LudwigModel(
            config=config_path,
            logging_level=logging.INFO,
            callbacks=[progress_callback],
        )

        # Run training in a thread to keep the event loop responsive.
        # asyncio.to_thread copies the current context (incl. the active
        # span) into that thread, so the epoch-end callback below can still
        # attach its span events to `ludwig.train`.
        with tracer.start_as_current_span("ludwig.train"):
            await asyncio.to_thread(
                model.train,
                dataset=dataset_path,
                output_directory=output_path,
                experiment_name="results",
            )

        # Check if aborted
        if await nats_service.is_aborted(str(run_id)):
            logger.info(f"Training aborted for run {run_id}")
            await nats_service.publish_status(str(run_id), "canceled")
            return

        # Best-effort evaluation report (confusion matrix / per-class stats /
        # misclassified rows — see services/evaluate.py). Unlike export.py's
        # golden sample, a bug here must never fail a run that trained fine,
        # so every exception is caught, not just the environmental ones.
        try:
            eval_df = await asyncio.to_thread(pd.read_parquet, dataset_path)
            result = await asyncio.to_thread(
                build_evaluation_report, model, eval_df, SPLIT_COLUMN_NAME, ITEM_ID_COLUMN_NAME
            )
            if result is None:
                logger.info(f"No data to evaluate for run {run_id} — skipping evaluation report")
            else:
                report, predictions_df = result
                report_key = evaluation_report_key(run_id)
                predictions_key = evaluation_predictions_key(run_id)
                await asyncio.to_thread(upload_json, BUCKET_TRAINING, report_key, report)
                await asyncio.to_thread(
                    predictions_df.to_parquet, s3fs_readable_path(BUCKET_TRAINING, predictions_key)
                )
                overall = report.get("overall", {})
                headline = overall.get("accuracy") if report["outputType"] == "category" else overall.get("r2")
                await nats_service.publish_evaluation_event(
                    str(run_id),
                    "success",
                    split=report["split"],
                    report_key=report_key,
                    predictions_key=predictions_key,
                    headline_metric=headline,
                )
        except Exception as e:
            logger.exception(f"Evaluation report failed for run {run_id} — training result is unaffected")
            await nats_service.publish_evaluation_event(str(run_id), "failed", error=str(e))

        await nats_service.publish_status(str(run_id), "succeeded")

        logger.info(f"Training completed for run {run_id}")

    except KeyboardInterrupt:
        logger.info(f"Training aborted for run {run_id}")
        await nats_service.publish_status(str(run_id), "canceled")

    except Exception:
        # No terminal "failed" event here — that used to fire on every retry
        # attempt (redundant). Now it's published exactly once, by
        # `on_train_permanent_failure` below, once the NATS consume loop has
        # exhausted all delivery attempts (see services/nats.py `_consume_loop`).
        logger.exception(f"Training failed for run {run_id}")
        raise  # Let NATS nak the message for retry

    finally:
        root_logger.removeHandler(log_handler)
        await log_handler.aclose()
        if local_log_path.exists():
            try:
                await asyncio.to_thread(
                    upload_file,
                    BUCKET_TRAINING,
                    training_logs_key(run_id),
                    str(local_log_path),
                )
            except Exception:
                logger.warning(
                    f"Failed to upload training log for run {run_id}", exc_info=True
                )
            local_log_path.unlink(missing_ok=True)


async def on_train_permanent_failure(data: TrainTask, error: str) -> None:
    """Called by the NATS consume loop once a `TrainTask` has exhausted all
    delivery attempts (see `services/nats.py` `_consume_loop`'s
    `on_permanent_failure`). Publishes the terminal 'failed' status event
    exactly once — `handle_train`'s own exception handler no longer does
    this itself, since it used to fire on every retry attempt."""
    logger.error(f"Training permanently failed for run {data.run_id} after retries: {error}")
    await nats_service.publish_status(str(data.run_id), "failed", message=error)
