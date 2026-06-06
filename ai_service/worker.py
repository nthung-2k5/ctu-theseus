"""
Celery worker for distributed AI training tasks using Ludwig.

Start workers:
    celery -A ai_service.worker worker --loglevel=info --concurrency=1

For multiple worker nodes, run the above command on each machine,
all pointing to the same Redis broker.
"""

from ludwig.utils.types import DataFrame
import json
import logging
import os

import pandas as pd
import requests
from celery import Celery
from celery.contrib.abortable import AbortableTask
from ludwig.api import LudwigModel
from ludwig.callbacks import Callback

from ai_service.dirs import DATASET_DIR, MODEL_DIR
from ai_service.dto import TrainRequest
from ai_service.models import get_model_config, get_model_meta, has_model

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:3000")

celery_app = Celery("theseus_ai", broker=f"{REDIS_URL}/0", backend=f"{REDIS_URL}/1")
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


def _send_webhook(webhook_url: str, payload: dict):
    """Send a webhook notification, logging but swallowing any errors."""
    try:
        requests.post(f"{GATEWAY_URL}{webhook_url}", json=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Webhook delivery failed: {e}")


# ──────────────────────────────────────────────────────────────────
# Ludwig Callback for real-time progress reporting
# ──────────────────────────────────────────────────────────────────

class TrainingProgressCallback(Callback):
    """Ludwig callback that reports per-epoch metrics via Celery state and webhooks."""

    def __init__(self, celery_task, webhook_url: str | None, total_epochs: int):
        self.celery_task = celery_task
        self.webhook_url = webhook_url
        self.total_epochs = total_epochs

    def on_epoch_end(self, trainer, progress_tracker, save_path, **kwargs):
        """Called by Ludwig at the end of each training epoch."""
        # Check for abort
        if hasattr(self.celery_task, 'is_aborted') and self.celery_task.is_aborted():
            raise KeyboardInterrupt("Training aborted by user")

        epoch = progress_tracker.epoch

        # Extract metrics from progress_tracker
        train_metrics = progress_tracker.train_metrics
        val_metrics = progress_tracker.validation_metrics

        # Ludwig stores metrics as dict[feature_name][metric_name] = list of values
        # For category output, we look for 'loss' and 'accuracy'
        train_loss = 0.0
        val_loss = 0.0
        accuracy = 0.0

        # Get the output feature name (typically 'label' or 'class')
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

        state_meta = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'accuracy': float(accuracy),
            'mAP': float(accuracy),  # Ludwig doesn't compute mAP by default; using accuracy as proxy
        }

        # Update Celery state so the API can poll the status
        self.celery_task.update_state(state='TRAINING', meta=state_meta)

        if self.webhook_url:
            _send_webhook(self.webhook_url, {
                "task_id": self.celery_task.request.id, "status": "training", "metrics": state_meta
            })


# ──────────────────────────────────────────────────────────────────
# Helper: Build Ludwig config from TrainRequest
# ──────────────────────────────────────────────────────────────────

def _build_ludwig_config(req: TrainRequest) -> dict:
    """Convert a TrainRequest into a Ludwig configuration dictionary."""
    model_config = req.model
    opt_config = req.optimization
    sched_config = req.schedule

    # Get the Ludwig encoder config for this architecture
    encoder_config = get_model_config(model_config.architecture)

    # Add pretrained and dropout settings
    encoder_config["pretrained"] = model_config.pretrained
    if model_config.drop_rate > 0:
        encoder_config["drop_rate"] = model_config.drop_rate

    # Map optimizer name to Ludwig format
    optimizer_type = opt_config.optimizer
    if optimizer_type == "adamw":
        optimizer_type = "adamw"
    elif optimizer_type == "adam":
        optimizer_type = "adam"
    elif optimizer_type == "sgd":
        optimizer_type = "sgd"

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
            "epochs": sched_config.epochs,
            "batch_size": req.dataset.batch_size,
            "optimizer": {
                "type": optimizer_type,
                "learning_rate": opt_config.learning_rate,
                "weight_decay": opt_config.weight_decay if optimizer_type != "sgd" else 0.0,
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


@celery_app.task(bind=True, base=AbortableTask)
def train_model_task(self, model_id: str, payload: dict):
    """Celery Task to execute training using Ludwig."""

    # Deserialize the JSON dict back into a validated Pydantic model
    req = TrainRequest(**payload)
    webhook_url = req.webhook_url
    task_id = self.request.id

    # 0. Sending start notification
    if webhook_url:
        _send_webhook(webhook_url, { "task_id": task_id, "status": "training" })

    try:
        # 1. Build dataset DataFrame
        dataset_name = req.dataset.source_uri.strip('/').split('/')[-1]
        base_dataset_dir = os.path.join(DATASET_DIR, dataset_name)
        dataset_df = _build_dataset_df(base_dataset_dir)

        if dataset_df.empty:
            raise ValueError(f"No images found in dataset directory: {base_dataset_dir}")

        # Extract class mapping from the dataset
        unique_classes = sorted(dataset_df['label'].unique().tolist())
        class_mapping = {name: idx for idx, name in enumerate(unique_classes)}

        # 2. Build Ludwig config
        ludwig_config = _build_ludwig_config(req)

        # 3. Create Ludwig model with progress callback
        progress_callback = TrainingProgressCallback(
            celery_task=self,
            webhook_url=webhook_url,
            total_epochs=req.schedule.epochs,
        )

        model = LudwigModel(
            config=ludwig_config,
            logging_level=logging.INFO,
            callbacks=[progress_callback],
        )

        # 4. Train the model
        model_save_dir = os.path.join(MODEL_DIR, model_id)
        os.makedirs(model_save_dir, exist_ok=True)

        train_stats, _, output_directory = model.train(
            dataset=dataset_df,
            output_directory=model_save_dir,
        )

        if self.is_aborted():
            return

        # 5. Save class mapping alongside the model
        map_path = os.path.join(model_save_dir, "class_mapping.json")
        with open(map_path, 'w') as f:
            json.dump(class_mapping, f)

        final_result = {
            "weights": output_directory,
            "mapping": map_path,
        }

        # 6. Sending completed notification
        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": task_id, "status": "completed", "result": final_result
            })

        return final_result

    except KeyboardInterrupt:
        # Raised by callback when task is aborted
        logger.info(f"Training aborted for model {model_id}")
        return

    except Exception as e:
        logger.exception(f"Training failed for model {model_id}")
        if webhook_url:
            # 6.1. Sending failed notification
            _send_webhook(webhook_url, {
                "task_id": task_id, "status": "failed", "error": str(e)
            })
        raise


@celery_app.task(bind=True)
def inference_task(self, model_id: str, model_name: str, image_path: str, threshold: float = 0.5, webhook_url: str | None = None):
    """Celery Task for inference using a saved Ludwig model."""

    try:
        model_save_dir = os.path.join(MODEL_DIR, model_id)
        map_path = os.path.join(model_save_dir, "class_mapping.json")

        # Load the class mapping JSON
        with open(map_path, 'r') as f:
            class_mapping = json.load(f)

        # Find the Ludwig model output directory (contains model artifacts)
        # Ludwig saves to output_directory/experiment_run/model/
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if 'model_hyperparameters.json' in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            # Try the direct path
            ludwig_model_dir = model_save_dir

        # Load the trained Ludwig model
        model = LudwigModel.load(ludwig_model_dir)

        # Prepare input as a DataFrame
        input_df = pd.DataFrame({'image_path': [image_path]})

        # Run inference
        predictions, _ = model.predict(dataset=input_df)

        assert isinstance(predictions, DataFrame)

        # Parse predictions into threshold-filtered results
        results = {}
        # Ludwig returns columns like 'label_predictions', 'label_probabilities', etc.
        if 'label_predictions' in predictions.columns:
            predicted_label = predictions['label_predictions'].iloc[0]

        # Get probabilities for all classes if available
        if 'label_probabilities' in predictions.columns:
            probs = predictions['label_probabilities'].iloc[0]
            if isinstance(probs, list):
                # Map probabilities to class names
                idx_to_class = {v: k for k, v in class_mapping.items()}
                for idx, prob in enumerate(probs):
                    if prob >= threshold and idx in idx_to_class:
                        results[idx_to_class[idx]] = round(float(prob), 4)
            elif isinstance(probs, dict):
                for class_name, prob in probs.items():
                    if prob >= threshold:
                        results[class_name] = round(float(prob), 4)

        # If no probability breakdown available, use the top prediction
        if not results and 'label_predictions' in predictions.columns:
            predicted_label = predictions['label_predictions'].iloc[0]
            results[str(predicted_label)] = 1.0

        # Sort results by confidence descending
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": self.request.id, "status": "success", "result": results
            })

        return results

    except Exception as e:
        logger.exception(f"Inference failed for model {model_id}")
        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": self.request.id, "status": "failed", "error": str(e)
            })
        raise
    finally:
        # Clean up uploaded image file to prevent disk fill
        if os.path.exists(image_path):
            os.remove(image_path)


@celery_app.task(bind=True)
def export_model_task(self, model_id: str, model_name: str, num_classes: int, export_format: str, webhook_url: str | None = None):
    """Celery Task to export the model using Ludwig."""

    try:
        model_save_dir = os.path.join(MODEL_DIR, model_id)
        export_path = os.path.join(MODEL_DIR, f"{model_id}.{export_format}")

        # Find the Ludwig model output directory
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if 'model_hyperparameters.json' in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            ludwig_model_dir = model_save_dir

        # Load the trained Ludwig model
        model = LudwigModel.load(ludwig_model_dir)

        # Export using Ludwig's export functionality
        if export_format == "torchscript":
            model.export_model(export_path, format="torch_export")
        elif export_format == "onnx":
            model.export_model(export_path, format="onnx")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        final_result = {"status": "success", "format": export_format, "path": export_path}

        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": self.request.id, "status": "success", "result": final_result
            })

        return final_result

    except Exception as e:
        logger.exception(f"Export failed for model {model_id}")
        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": self.request.id, "status": "failed", "error": str(e)
            })
        raise
