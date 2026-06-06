"""
Celery worker for distributed AI training tasks.

Start workers:
    celery -A ai_service.worker worker --loglevel=info --concurrency=1

For multiple worker nodes, run the above command on each machine,
all pointing to the same Redis broker.
"""

import json
import logging
import os

import requests
from celery import Celery
from celery.contrib.abortable import AbortableTask
from timm.data import create_dataset, create_loader

from ai_service.dirs import DATASET_DIR, MODEL_DIR
from ai_service.dto import TrainRequest
from ai_service.models import get_model

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


@celery_app.task(bind=True, base=AbortableTask)
def train_model_task(self, model_id: str, payload: dict):
    """Celery Task to execute the full training loop."""

    # Deserialize the JSON dict back into a validated Pydantic model
    req = TrainRequest(**payload)
    ds_config = req.dataset
    model_config = req.model
    opt_config = req.optimization
    sched_config = req.schedule
    adv_config = req.advanced_features
    webhook_url = req.webhook_url
    task_id = self.request.id

    # 0. Sending start notification
    if webhook_url:
        _send_webhook(webhook_url, { "task_id": task_id, "status": "training" })

    try:
        # 1. Setup Datasets & Loaders
        dataset_name = ds_config.source_uri.strip('/').split('/')[-1]
        base_dataset_dir = os.path.join(DATASET_DIR, dataset_name)
        dataset_train = create_dataset(
            name='',
            root=base_dataset_dir,
            split='train',
            is_training=True
        )
        dataset_val = create_dataset(
            name='',
            root=base_dataset_dir,
            split='validation',
            is_training=False
        )

        class_mapping = dataset_train.reader.class_to_idx
        num_classes = len(class_mapping)

        # 2. Build Model via Registry
        ModelClass = get_model(model_config.architecture)
        worker = ModelClass(
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            drop_rate=model_config.drop_rate
        )

        train_loader = create_loader(
            dataset_train, input_size=worker.data_config['input_size'],
            batch_size=ds_config.batch_size, is_training=True, num_workers=ds_config.num_workers
        )
        val_loader = create_loader(
            dataset_val, input_size=worker.data_config['input_size'],
            batch_size=ds_config.batch_size, is_training=False, num_workers=ds_config.num_workers
        )

        # 3. Setup Training with all user-specified parameters
        worker.setup_training(
            opt_name=opt_config.optimizer,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            sched_name=sched_config.scheduler,
            epochs=sched_config.epochs,
            warmup_epochs=sched_config.warmup_epochs,
            use_ema=adv_config.use_ema,
            mixup_alpha=adv_config.mixup_alpha,
            cutmix_alpha=adv_config.cutmix_alpha
        )

        # 4. Training Loop
        for epoch in range(sched_config.epochs):
            if self.is_aborted():
                return

            train_loss = worker.train_one_epoch(train_loader, epoch)

            if self.is_aborted():
                return

            metrics = worker.validate(val_loader)

            if self.is_aborted():
                return

            state_meta = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': metrics['val_loss'],
                'accuracy': metrics['accuracy'],
                'mAP': metrics['mAP']
            }

            # Update Celery state so the API can poll the status
            self.update_state(state='TRAINING', meta=state_meta)

            if self.is_aborted():
                return

            if webhook_url:
                _send_webhook(webhook_url, {
                    "task_id": task_id, "status": "training", "metrics": state_meta
                })

            if self.is_aborted():
                return

        # 5. Save to Disk
        weight_path, map_path = worker.save_model(model_id, class_mapping)
        final_result = {"weights": weight_path, "mapping": map_path}

        # 6. Sending completed notification
        if webhook_url:
            _send_webhook(webhook_url, {
                "task_id": task_id, "status": "completed", "result": final_result
            })

        return final_result

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
    """Celery Task for inference using saved weights."""

    try:
        weight_path = os.path.join(MODEL_DIR, f"{model_id}.pt")
        map_path = os.path.join(MODEL_DIR, f"{model_id}_map.json")

        # Load the class mapping JSON
        with open(map_path, 'r') as f:
            class_mapping = json.load(f)

        num_classes = len(class_mapping)

        # Initialize model via registry and load saved weights
        ModelClass = get_model(model_name)
        worker = ModelClass(
            num_classes=num_classes,
            weights_path=weight_path
        )

        # Run inference with threshold
        results = worker.inference(image_path, class_mapping, threshold)

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
    """Celery Task to export the model on-demand."""

    try:
        weight_path = os.path.join(MODEL_DIR, f"{model_id}.pt")
        export_path = os.path.join(MODEL_DIR, f"{model_id}.{export_format}")

        # Initialize model via registry and load saved weights
        ModelClass = get_model(model_name)
        worker = ModelClass(
            num_classes=num_classes,
            weights_path=weight_path
        )

        worker.export(export_format, export_path)
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
