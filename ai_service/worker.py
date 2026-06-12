"""
NATS JetStream worker for distributed AI tasks.

Replaces the Celery worker. Consumes task messages from NATS JetStream,
routes them to the appropriate handler (train/inference/export), and
publishes results and progress back to NATS.

Start the worker:
    python -m ai_service.worker
"""

from pydantic import ConfigDict, BaseModel
from pydantic.alias_generators import to_camel
from nats.aio.msg import Msg
from ludwig.utils.types import DataFrame
import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback

from ai_service.models import get_model_config, get_model_meta, has_model
from ai_service.nats_client import nats_service
from ai_service.storage import (
    BUCKET_EXPORTS,
    BUCKET_MODELS,
    cleanup_temp,
    download_dataset,
    download_model,
    ensure_buckets,
    upload_directory,
    upload_file,
    upload_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
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
            if 'loss' in feature_metrics:
                val_loss = feature_metrics['loss'][-1] if isinstance(feature_metrics['loss'], list) else feature_metrics['loss']
            if 'accuracy' in feature_metrics:
                accuracy = feature_metrics['accuracy'][-1] if isinstance(feature_metrics['accuracy'], list) else feature_metrics['accuracy']

        for feature_name in train_metrics:
            feature_metrics = train_metrics[feature_name]
            if 'loss' in feature_metrics:
                train_loss = feature_metrics['loss'][-1] if isinstance(feature_metrics['loss'], list) else feature_metrics['loss']

        metrics = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'accuracy': float(accuracy),
            'mAP': float(accuracy),  # Ludwig doesn't compute mAP by default; using accuracy as proxy
        }

        # Publish progress to NATS (from sync callback via event loop)
        future = asyncio.run_coroutine_threadsafe(
            nats_service.publish_progress(self.run_id, {
                "id": self.run_id,
                "epoch": epoch,
                "metrics": metrics,
            }),
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
                "weight_decay": opt_config.get("weight_decay", 0.05) if optimizer_type != "sgd" else 0.0,
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
    for split_name in ['train', 'validation']:
        split_dir = os.path.join(dataset_dir, split_name)
        if not os.path.isdir(split_dir):
            continue

        # Ludwig uses 0 for train, 1 for validation, 2 for test
        split_value = 0 if split_name == 'train' else 1

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                if os.path.isfile(filepath):
                    rows.append({
                        'image_path': filepath,
                        'label': class_name,
                        'split': split_value,
                    })

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
    await nats_service.publish_result("train", run_id, {
        "id": run_id,
        "type": "train",
        "status": "training",
        "completed_at": None,
    })

    try:
        # 1. Download dataset from S3 to local temp dir
        project_id = data.get("project_id", "")
        local_dataset_dir = download_dataset(project_id, run_id)
        dataset_df = _build_dataset_df(local_dataset_dir)

        if dataset_df.empty:
            raise ValueError(f"No images found in downloaded dataset for run {run_id}")

        # Extract class mapping from the dataset
        unique_classes = sorted(dataset_df['label'].unique().tolist())
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
        model_save_dir = os.path.join(tempfile.gettempdir(), "theseus", "training", run_id)
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
        with open(map_path, 'w') as f:
            json.dump(class_mapping, f)

        # 6. Upload trained model to S3
        upload_model(run_id, model_save_dir)

        # 7. Publish completion result
        await nats_service.publish_result("train", run_id, {
            "id": run_id,
            "type": "train",
            "status": "completed",
            "result": {
                "model_key": f"{run_id}/",
                "mapping_key": f"{run_id}/class_mapping.json",
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"Training completed for run {run_id}")

        # 8. Clean up local temp files
        cleanup_temp("datasets", run_id)

    except KeyboardInterrupt:
        logger.info(f"Training aborted for run {run_id}")
        _abort_requests.discard(run_id)

    except Exception as e:
        logger.exception(f"Training failed for run {run_id}")
        await nats_service.publish_result("train", run_id, {
            "id": run_id,
            "type": "train",
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise  # Let NATS nak the message for retry

class InferenceRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    upload_key: str
    upload_filename: str
    threshold: float = 0.5


async def handle_inference(msg: Msg) -> None:
    """Handle an inference task message."""
    request = InferenceRequest.model_validate_json(msg.data.decode("utf-8"))
    model_id = msg.subject.split(".")[-1]

    logger.info(f"Starting inference job for model {model_id}")

    try:
        # 1. Download model from S3 (cached locally)
        model_save_dir = download_model(model_id)
        map_path = os.path.join(model_save_dir, "class_mapping.json")

        with open(map_path, 'r') as f:
            class_mapping = json.load(f)

        # Find the Ludwig model output directory
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if 'model_hyperparameters.json' in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            ludwig_model_dir = model_save_dir

        # 2. Download inference input image from NATS Object Store
        upload_data = await nats_service.get_upload(request.upload_key)
        if upload_data is None:
            raise FileNotFoundError(f"Upload not found for key: {request.upload_key}")

        import tempfile
        local_image_dir = os.path.join(tempfile.gettempdir(), "theseus", "uploads", job_id)
        os.makedirs(local_image_dir, exist_ok=True)
        local_image_path = os.path.join(local_image_dir, request.upload_filename)

        with open(local_image_path, "wb") as f:
            f.write(upload_data)

        # 3. Load and run inference in a thread
        def _run_inference():
            model = LudwigModel.load(ludwig_model_dir)
            input_df = pd.DataFrame({'image_path': [local_image_path]})
            predictions, _ = model.predict(dataset=input_df)
            return predictions

        predictions = await asyncio.to_thread(_run_inference)

        assert isinstance(predictions, DataFrame)

        # Parse predictions into threshold-filtered results
        results = {}
        if 'label_probabilities' in predictions.columns:
            probs = predictions['label_probabilities'].iloc[0]
            if isinstance(probs, list):
                idx_to_class = {v: k for k, v in class_mapping.items()}
                for idx, prob in enumerate(probs):
                    if prob >= request.threshold and idx in idx_to_class:
                        results[idx_to_class[idx]] = round(float(prob), 4)
            elif isinstance(probs, dict):
                for class_name, prob in probs.items():
                    if prob >= request.threshold:
                        results[class_name] = round(float(prob), 4)

        if not results and 'label_predictions' in predictions.columns:
            predicted_label = predictions['label_predictions'].iloc[0]
            results[str(predicted_label)] = 1.0

        # Sort results by confidence descending
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        # Publish result
        await msg.respond(json.dumps({
            "status": "success",
            "results": results,
        }).encode("utf-8"))

        logger.info(f"Inference completed for job {model_id}")

    except Exception as e:
        logger.exception(f"Inference failed for job {model_id}")
        await msg.respond(json.dumps({
            "status": "failed",
            "error": str(e),
        }).encode("utf-8"))
        raise
    finally:
        # Clean up local temp files
        cleanup_temp("uploads", model_id)


async def handle_export(data: dict[str, Any]) -> None:
    """Handle an export task message."""
    job_id = data["id"]
    model_id = data["model_id"]
    export_format = data.get("export_format", "onnx")

    logger.info(f"Starting export job {job_id} for model {model_id} (format: {export_format})")

    try:
        # 1. Download model from S3 (cached locally)
        model_save_dir = download_model(model_id)

        # Find the Ludwig model output directory
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if 'model_hyperparameters.json' in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            ludwig_model_dir = model_save_dir

        import tempfile
        export_path = os.path.join(tempfile.gettempdir(), "theseus", "exports", job_id, f"model.{export_format}")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        def _run_export():
            model = LudwigModel.load(ludwig_model_dir)
            if export_format == "torchscript":
                model.export_model(export_path, format="torch_export")
            elif export_format == "onnx":
                model.export_model(export_path, format="onnx")
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            return export_path

        result_path = await asyncio.to_thread(_run_export)

        # 2. Upload exported model to S3
        export_s3_key = f"{model_id}/model.{export_format}"
        upload_file(BUCKET_EXPORTS, export_s3_key, result_path)

        await nats_service.publish_result("export", job_id, {
            "id": job_id,
            "type": "export",
            "status": "success",
            "result": {
                "format": export_format,
                "export_key": export_s3_key,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"Export completed for job {job_id}")

        # Clean up local temp
        cleanup_temp("exports", job_id)

    except Exception as e:
        logger.exception(f"Export failed for job {job_id}")
        await nats_service.publish_result("export", job_id, {
            "id": job_id,
            "type": "export",
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

async def main():
    """Start the NATS worker: connect, subscribe to all task subjects, and process."""
    await nats_service.connect()

    # Ensure S3 buckets exist
    ensure_buckets()

    logger.info("AI Worker started. Listening for tasks...")

    # Run task consumers and command consumer concurrently
    await asyncio.gather(
        nats_service.subscribe_tasks("theseus.tasks.train.*", "train-worker", handle_train),
        nats_service.subscribe("theseus.inference.*", handle_inference),
        nats_service.subscribe_tasks("theseus.tasks.export.*", "export-worker", handle_export),
        nats_service.subscribe_commands("theseus.commands.>", handle_command),
    )


def _shutdown(loop: asyncio.AbstractEventLoop):
    """Handle graceful shutdown."""
    logger.info("Shutting down AI Worker...")
    loop.create_task(nats_service.close())


if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    # Handle SIGINT/SIGTERM for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: _shutdown(loop))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _shutdown(loop))

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted.")
    finally:
        loop.run_until_complete(nats_service.close())
        loop.close()
