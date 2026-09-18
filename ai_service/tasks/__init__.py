import asyncio
import logging

from schema.command import Command
from schema.export_task import ExportTask
from schema.inference_task import InferenceTask
from schema.train_task import TrainTask
from services.nats import nats_service
from services.storage import ensure_buckets

from schema import subjects as subj

from .export import handle_export, on_export_permanent_failure
from .inference import handle_inference_task, handle_inference_warm, on_inference_permanent_failure
from .train import handle_command, handle_train, on_train_permanent_failure

__all__ = ["run_task_workers"]

logger = logging.getLogger(__name__)

# Shared queue group for the fire-and-forget warm-cache subscription: if
# this worker is ever scaled to multiple replicas, only one replica handles
# each warm signal instead of every replica receiving every message (a bare
# core-NATS `subscribe()` with no queue group fans out rather than
# load-balancing). The JetStream pull consumers below already load-balance
# correctly across replicas via their shared `durable_name`, so they don't
# need one.
INFERENCE_QUEUE_GROUP = "inference-workers"


async def run_task_workers():
    await nats_service.connect()

    # Ensure S3 buckets exist
    ensure_buckets()

    logger.info("AI Worker started. Listening for tasks...")

    # Run task consumers and command consumer concurrently
    await asyncio.gather(
        nats_service.subscribe_tasks(
            subj.SUBJECT_WILDCARDS["trainTasks"],
            "train-worker",
            handle_train,
            TrainTask,
            on_permanent_failure=on_train_permanent_failure,
            # No redelivery for training. A retry is a full GPU retrain, and the
            # failures that actually happen here (OOM, a bad config, a dataset
            # the model can't fit) are deterministic — retrying burns hours to
            # reach the same result. A failed run is reported terminally and the
            # user retrains explicitly.
            max_deliver=1,
        ),
        nats_service.subscribe_tasks(
            subj.SUBJECT_WILDCARDS["inferenceTasks"],
            "inference-worker",
            handle_inference_task,
            InferenceTask,
            on_permanent_failure=on_inference_permanent_failure,
            nak_delay=30,
        ),
        nats_service.subscribe(
            subj.SUBJECT_WILDCARDS["inferenceWarmAll"],
            handle_inference_warm,
            queue=INFERENCE_QUEUE_GROUP,
        ),
        nats_service.subscribe_tasks(
            subj.SUBJECT_WILDCARDS["exportTasks"],
            "export-worker",
            handle_export,
            ExportTask,
            on_permanent_failure=on_export_permanent_failure,
            nak_delay=30,
        ),
        nats_service.subscribe_commands(
            subj.SUBJECT_WILDCARDS["commandsPerRun"], handle_command, Command
        ),
    )


async def close_task_workers():
    await nats_service.close()
