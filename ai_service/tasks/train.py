import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Literal
from uuid import UUID

import ludwig.constants as ludwig_consts
from config import TEMP_DIR
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker
from opentelemetry import trace
from schema.command import Command
from schema.train_task import TrainTask
from services.nats import nats_service
from services.storage import (
    BUCKET_DATASETS,
    BUCKET_TRAINING,
    file_exists,
    s3fs_readable_path,
    training_logs_key,
    upload_file,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("theseus-worker")

# ──────────────────────────────────────────────────────────────────
# Abort tracking
# ──────────────────────────────────────────────────────────────────

# Set of run IDs that have been requested to abort
_abort_requests: set[str] = set()


async def handle_command(data: Command) -> None:
    """Handle a command message (stop/abort)."""
    logger.info(f"Received abort command for run {data.run_id}")
    _abort_requests.add(str(data.run_id))


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
        # Check for abort
        if str(self.run_id) in _abort_requests:
            _abort_requests.discard(str(self.run_id))
            raise KeyboardInterrupt("Training aborted by user")

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

    local_log_path = TEMP_DIR / "logs" / f"{run_id}.log"
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handler = RunLogHandler(run_id, loop, local_log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    # Notify that training has started
    await nats_service.publish_status(str(run_id), "running")

    try:
        if not file_exists(BUCKET_TRAINING, data.config_key):
            raise ValueError(
                f"No training config found at {data.config_key} for run {run_id}"
            )

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
        if str(run_id) in _abort_requests:
            _abort_requests.discard(str(run_id))
            logger.info(f"Training aborted for run {run_id}")
            await nats_service.publish_status(str(run_id), "canceled")
            return

        await nats_service.publish_status(str(run_id), "succeeded")

        logger.info(f"Training completed for run {run_id}")

    except KeyboardInterrupt:
        logger.info(f"Training aborted for run {run_id}")
        _abort_requests.discard(str(run_id))
        await nats_service.publish_status(str(run_id), "canceled")

    except Exception as e:
        logger.exception(f"Training failed for run {run_id}")
        await nats_service.publish_status(str(run_id), "failed", message=str(e))
        raise  # Let NATS nak the message for retry

    finally:
        root_logger.removeHandler(log_handler)
        await log_handler.aclose()
        if local_log_path.exists():
            try:
                upload_file(
                    BUCKET_TRAINING, training_logs_key(run_id), str(local_log_path)
                )
            except Exception:
                logger.warning(
                    f"Failed to upload training log for run {run_id}", exc_info=True
                )
            local_log_path.unlink(missing_ok=True)
