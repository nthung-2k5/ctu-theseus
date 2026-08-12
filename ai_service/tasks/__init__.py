import asyncio
import logging

from ai_service.schema.command import Command
from ai_service.schema.export_task import ExportTask
from ai_service.schema.train_task import TrainTask
from ai_service.services.nats import nats_service
from ai_service.services.storage import ensure_buckets

from .export import handle_export
from .inference import handle_inference
from .train import handle_command, handle_train

__all__ = ["run_task_workers"]

logger = logging.getLogger(__name__)


async def run_task_workers():
    await nats_service.connect()

    # Ensure S3 buckets exist
    ensure_buckets()

    logger.info("AI Worker started. Listening for tasks...")

    # Run task consumers and command consumer concurrently
    await asyncio.gather(
        nats_service.subscribe_tasks(
            "theseus.task.train.*", "train-worker", handle_train, TrainTask
        ),
        nats_service.subscribe("theseus.inference.*", handle_inference),
        nats_service.subscribe_tasks(
            "theseus.task.export.*", "export-worker", handle_export, ExportTask
        ),
        nats_service.subscribe_commands(
            "theseus.command.run.*", handle_command, Command
        ),
    )


async def close_task_workers():
    await nats_service.close()
