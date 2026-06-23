"""Task workers entry point"""

from ai_service.schema.training import LudwigTrainingConfiguration
import asyncio
import logging

from ai_service.models import list_models
from ai_service.services.nats import nats_service
from ai_service.services.storage import ensure_buckets

from .export import handle_export
from .inference import handle_inference
from .train import handle_command, handle_train

__all__ = ["run_task_workers"]

logger = logging.getLogger(__name__)


async def run_task_workers():
    """Start the NATS worker: connect, subscribe to all task subjects, and process."""
    await nats_service.connect()

    # Ensure S3 buckets exist
    ensure_buckets()

    logger.info("AI Worker started. Listening for tasks...")

    # Run task consumers and command consumer concurrently
    await asyncio.gather(
        nats_service.subscribe_tasks(
            "theseus.tasks.train.*", "train-worker", handle_train, LudwigTrainingConfiguration
        ),
        nats_service.subscribe("theseus.inference.*", handle_inference),
        nats_service.subscribe(
            "theseus.models",
            lambda msg: nats_service.publish("theseus.models.response", list_models()),
        ),
        nats_service.subscribe_tasks(
            "theseus.tasks.export.*", "export-worker", handle_export
        ),
        nats_service.subscribe_commands("theseus.commands.>", handle_command),
    )


async def close_task_workers():
    await nats_service.close()
