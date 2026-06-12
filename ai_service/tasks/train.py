import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback

from ai_service.models import get_model_config
from ai_service.services.nats import nats_service
from ai_service.services.storage import cleanup_temp, download_dataset, upload_model

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

    def __init__(self, run_id: str, loop: asyncio.AbstractEventLoop):
        self.run_id = run_id
        self.loop = loop

    def on_epoch_end(self, trainer, progress_tracker, save_path, **kwargs):
        """Called by Ludwig at the end of each training epoch."""
        # Check for abort
        if self.run_id in _abort_requests:
            _abort_requests.discard(self.run_id)
            raise KeyboardInterrupt("Training aborted by user")

        epoch = progress_tracker.epoch

        # Extract metrics from progress_tracker
        train_metrics = progress_tracker.train_metrics
        val_metrics = progress_tracker.validation_metrics

        train_loss = 0.0
        val_loss = 0.0
        accuracy = 0.0

        for feature_name in val_metrics:
            feature_metrics = val_metrics[feature_name]
            if "loss" in feature_metrics:
                val_loss = (
                    feature_metrics["loss"][-1]
                    if isinstance(feature_metrics["loss"], list)
                    else feature_metrics["loss"]
                )
            if "accuracy" in feature_metrics:
                accuracy = (
                    feature_metrics["accuracy"][-1]
                    if isinstance(feature_metrics["accuracy"], list)
                    else feature_metrics["accuracy"]
                )

        for feature_name in train_metrics:
            feature_metrics = train_metrics[feature_name]
            if "loss" in feature_metrics:
                train_loss = (
                    feature_metrics["loss"][-1]
                    if isinstance(feature_metrics["loss"], list)
                    else feature_metrics["loss"]
                )

        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "accuracy": float(accuracy),
            "mAP": float(
                accuracy
            ),  # Ludwig doesn't compute mAP by default; using accuracy as proxy
        }

        # Publish progress to NATS (from sync callback via event loop)
        future = asyncio.run_coroutine_threadsafe(
            nats_service.publish_progress(
                self.run_id,
                {
                    "id": self.run_id,
                    "epoch": epoch,
                    "metrics": metrics,
                },
            ),
            self.loop,
        )
        # Wait for publish to complete (with timeout to not block training)
        try:
            future.result(timeout=5)
        except Exception as e:
            logger.warning(f"Failed to publish progress: {e}")


# ──────────────────────────────────────────────────────────────────
# Helper: Build Ludwig config from task payload
# ──────────────────────────────────────────────────────────────────


def _build_ludwig_config(payload: dict[str, Any]) -> dict:
    """Convert a task payload into a Ludwig configuration dictionary."""
    model_config = payload.get("model", {})
    opt_config = payload.get("optimization", {})
    sched_config = payload.get("schedule", {})
    dataset_config = payload.get("dataset", {})

    # Get the Ludwig encoder config for this architecture
    architecture = model_config.get("architecture", "resnet50")
    encoder_config = get_model_config(architecture)

    # Add pretrained and dropout settings
    encoder_config["pretrained"] = model_config.get("pretrained", True)
    drop_rate = model_config.get("drop_rate", 0.0)
    if drop_rate > 0:
        encoder_config["drop_rate"] = drop_rate

    # Map optimizer name to Ludwig format
    optimizer_type = opt_config.get("optimizer", "adamw")

    ludwig_config = {
        "input_features": [
            {
                "name": "image_path",
                "type": "image",
                "encoder": encoder_config,
            }
        ],
        "output_features": [
            {
                "name": "label",
                "type": "category",
            }
        ],
        "trainer": {
            "epochs": sched_config.get("epochs", 50),
            "batch_size": dataset_config.get("batch_size", 64),
            "optimizer": {
                "type": optimizer_type,
                "learning_rate": opt_config.get("learning_rate", 0.001),
                "weight_decay": opt_config.get("weight_decay", 0.05)
                if optimizer_type != "sgd"
                else 0.0,
            },
            "learning_rate_scheduler": {
                "type": "cosine",
                "warmup_fraction": 0.1,
            },
        },
        "preprocessing": {
            "split": {
                "type": "fixed",
                "column": "split",
            },
        },
    }

    return ludwig_config


# ──────────────────────────────────────────────────────────────────
# Helper: Build dataset DataFrame from the dataset directory
# ──────────────────────────────────────────────────────────────────


def _build_dataset_df(dataset_dir: str) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a dataset directory with structure:
        dataset_dir/
            train/
                class_a/
                    image1.jpg
                class_b/
                    image2.jpg
            validation/
                class_a/
                    image3.jpg
                class_b/
                    image4.jpg

    Returns a DataFrame with columns: image_path, label, split
    """
    rows = []
    for split_name in ["train", "validation"]:
        split_dir = os.path.join(dataset_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        # Ludwig uses 0 for train, 1 for validation, 2 for test
        split_value = 0 if split_name == "train" else 1

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                if os.path.isfile(filepath):
                    rows.append(
                        {
                            "image_path": filepath,
                            "label": class_name,
                            "split": split_value,
                        }
                    )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Task Handlers
# ──────────────────────────────────────────────────────────────────


async def handle_train(data: dict[str, Any]) -> None:
    """Handle a training task message."""
    run_id = data["id"]
    payload = data.get("payload", data)  # Support both wrapped and flat format
    loop = asyncio.get_event_loop()

    logger.info(f"Starting training for run {run_id}")

    # Notify that training has started
    await nats_service.publish_result(
        "train",
        run_id,
        {
            "id": run_id,
            "type": "train",
            "status": "training",
            "completed_at": None,
        },
    )

    try:
        # 1. Download dataset from S3 to local temp dir
        project_id = data.get("project_id", "")
        local_dataset_dir = download_dataset(project_id, run_id)
        dataset_df = _build_dataset_df(local_dataset_dir)

        if dataset_df.empty:
            raise ValueError(f"No images found in downloaded dataset for run {run_id}")

        # Extract class mapping from the dataset
        unique_classes = sorted(dataset_df["label"].unique().tolist())
        class_mapping = {name: idx for idx, name in enumerate(unique_classes)}

        # 2. Build Ludwig config
        ludwig_config = _build_ludwig_config(payload)

        # 3. Create Ludwig model with progress callback
        progress_callback = TrainingProgressCallback(run_id=run_id, loop=loop)

        model = LudwigModel(
            config=ludwig_config,
            logging_level=logging.INFO,
            callbacks=[progress_callback],
        )

        # 4. Train the model in a local temp dir
        import tempfile

        model_save_dir = os.path.join(
            tempfile.gettempdir(), "theseus", "training", run_id
        )
        os.makedirs(model_save_dir, exist_ok=True)

        # Run training in a thread to keep the event loop responsive
        train_stats, _, output_directory = await asyncio.to_thread(
            model.train,
            dataset=dataset_df,
            output_directory=model_save_dir,
        )

        # Check if aborted
        if run_id in _abort_requests:
            _abort_requests.discard(run_id)
            logger.info(f"Training aborted for run {run_id}")
            return

        # 5. Save class mapping alongside the model
        map_path = os.path.join(model_save_dir, "class_mapping.json")
        with open(map_path, "w") as f:
            json.dump(class_mapping, f)

        # 6. Upload trained model to S3
        upload_model(run_id, model_save_dir)

        # 7. Publish completion result
        await nats_service.publish_result(
            "train",
            run_id,
            {
                "id": run_id,
                "type": "train",
                "status": "completed",
                "result": {
                    "model_key": f"{run_id}/",
                    "mapping_key": f"{run_id}/class_mapping.json",
                },
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(f"Training completed for run {run_id}")

        # 8. Clean up local temp files
        cleanup_temp("datasets", run_id)

    except KeyboardInterrupt:
        logger.info(f"Training aborted for run {run_id}")
        _abort_requests.discard(run_id)

    except Exception as e:
        logger.exception(f"Training failed for run {run_id}")
        await nats_service.publish_result(
            "train",
            run_id,
            {
                "id": run_id,
                "type": "train",
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise  # Let NATS nak the message for retry
