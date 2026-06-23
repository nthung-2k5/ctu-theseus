import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import ludwig.constants as ludwig_consts
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import ProgressTracker

from ai_service.schema.training import (
    LudwigTrainingConfiguration,
)
from ai_service.schema.training_status import (
    TrainingCompletedStatus,
    TrainingFailedStatus,
    TrainingMetric,
    TrainingProgress,
    TrainingStartedStatus,
)
from ai_service.services.nats import nats_service
from ai_service.services.storage import (
    get_training_dataset_path,
    get_training_output_path,
    get_training_run_config_path,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Abort tracking
# ──────────────────────────────────────────────────────────────────

# Set of run IDs that have been requested to abort
_abort_requests: set[str] = set()


async def handle_command(data: dict[str, Any]) -> None:
    """Handle a command message (stop/abort)."""
    command = data.get("command")
    run_id = data.get("run_id")
    if command == "abort" and run_id:
        logger.info(f"Received abort command for run {run_id}")
        _abort_requests.add(run_id)


# ──────────────────────────────────────────────────────────────────
# Ludwig Callback for real-time progress reporting via NATS
# ──────────────────────────────────────────────────────────────────
class TrainingProgressCallback(Callback):
    """Ludwig callback that reports per-epoch metrics via NATS."""

    def __init__(self, run_id: UUID, loop: asyncio.AbstractEventLoop):
        self.run_id = run_id
        self.loop = loop

    def on_epoch_end(
        self, trainer, progress_tracker: ProgressTracker, save_path, **kwargs
    ):
        """Called by Ludwig at the end of each training epoch."""
        # Check for abort
        if self.run_id in _abort_requests:
            _abort_requests.discard(self.run_id)
            raise KeyboardInterrupt("Training aborted by user")

        epoch = progress_tracker.epoch

        # Extract metrics from progress_tracker
        train_metrics = progress_tracker.train_metrics[ludwig_consts.COMBINED]
        val_metrics = progress_tracker.validation_metrics[ludwig_consts.COMBINED]
        test_metrics = progress_tracker.test_metrics[ludwig_consts.COMBINED]

        def get_metric_for_split(
            split: dict[str, list[TrainerMetric]],
        ) -> TrainingMetric:
            return TrainingMetric(
                loss=float(split[ludwig_consts.LOSS][-1].value),
                accuracy=float(split[ludwig_consts.ACCURACY][-1].value),
            )

        # Publish progress to NATS (from sync callback via event loop)
        future = asyncio.run_coroutine_threadsafe(
            nats_service.publish_progress(
                self.run_id,
                TrainingProgress(
                    id=self.run_id,
                    epoch=epoch,
                    train_metrics=get_metric_for_split(train_metrics),
                    validation_metrics=get_metric_for_split(val_metrics),
                    test_metrics=get_metric_for_split(test_metrics),
                ),
            ),
            self.loop,
        )

        # Wait for publish to complete (with timeout to not block training)
        try:
            future.result(timeout=5)
        except Exception as e:
            logger.warning(f"Failed to publish progress: {e}")


# ──────────────────────────────────────────────────────────────────
# Task Handlers
# ──────────────────────────────────────────────────────────────────
async def handle_train(data: LudwigTrainingConfiguration) -> None:
    """Handle a training task message."""
    run_id = data.run_id
    dataset_version_id = data.dataset_version_id
    loop = asyncio.get_event_loop()

    logger.info(f"Starting training for run {run_id}")

    # Notify that training has started
    await nats_service.publish_status(
        "train",
        run_id,
        TrainingStartedStatus(
            id=run_id,
            type="train",
            status="training",
            timestamp=datetime.now(),
        ),
    )

    try:
        # 1. Check if training config on S3 exists
        config_path = get_training_run_config_path(run_id)

        if config_path is None:
            raise ValueError(f"No training config found for run {run_id}")

        # 3. Create Ludwig model with progress callback
        progress_callback = TrainingProgressCallback(run_id=run_id, loop=loop)

        model = LudwigModel(
            config=config_path,
            logging_level=logging.INFO,
            callbacks=[progress_callback],
        )

        # 4. Train the model
        # Run training in a thread to keep the event loop responsive
        train_stats, _, output_directory = await asyncio.to_thread(
            model.train,
            dataset=get_training_dataset_path(dataset_version_id),
            output_directory=get_training_output_path(run_id),
        )

        # Check if aborted
        if run_id in _abort_requests:
            _abort_requests.discard(run_id)
            logger.info(f"Training aborted for run {run_id}")
            return

        # 7. Publish completion result
        await nats_service.publish_status(
            "train",
            run_id,
            TrainingCompletedStatus(
                id=run_id,
                type="train",
                status="completed",
                timestamp=datetime.now(timezone.utc),
            ),
        )

        logger.info(f"Training completed for run {run_id}")

    except KeyboardInterrupt:
        logger.info(f"Training aborted for run {run_id}")
        _abort_requests.discard(run_id)

    except Exception as e:
        logger.exception(f"Training failed for run {run_id}")
        await nats_service.publish_status(
            "train",
            run_id,
            TrainingFailedStatus(
                id=run_id,
                type="train",
                status="failed",
                error_message=str(e),
                timestamp=datetime.now(timezone.utc),
            ),
        )
        raise  # Let NATS nak the message for retry
