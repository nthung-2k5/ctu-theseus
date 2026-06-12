"""Task workers entry point"""

import asyncio
import logging

from ai_service.services.nats import nats_service
from ai_service.services.storage import ensure_buckets
from ai_service.tasks.export import handle_export
from ai_service.tasks.inference import handle_inference
from ai_service.tasks.train import handle_command, handle_train

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
            "theseus.tasks.train.*", "train-worker", handle_train
        ),
        nats_service.subscribe("theseus.inference.*", handle_inference),
        nats_service.subscribe_tasks(
            "theseus.tasks.export.*", "export-worker", handle_export
        ),
        nats_service.subscribe_commands("theseus.commands.>", handle_command),
    )


async def close_task_workers():
    await nats_service.close()
