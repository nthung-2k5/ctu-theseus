"""Run one training job on this server: claim bookkeeping, live progress, cancel, evaluation row.

Framework-neutral since trainer backends became a plugin system: everything here is common to any
backend (Ludwig or otherwise) — the run lifecycle and abort registration. Training and evaluation themselves are
`run_core.execute`, which knows nothing about Postgres or the event writer, so a remote worker can run the same
code; this module is the local half that wires it to both. What changed relative to the pre-plugin worker
still applies:

  * Progress goes straight to the event writer (a thread-safe queue put) instead of a
    run_coroutine_threadsafe(...).result(timeout=5) round trip per publish.
  * Abort is a threading.Event check (see abort.py), not a JetStream KV read with a 5 s timeout.
  * There is no redelivery, so no on_permanent_failure callback and no "do not publish failed on
    every retry" dance: the run either finishes here or fails here, exactly once.
"""

import logging
import uuid
from dataclasses import dataclass

from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import TrainingRun
from omnitrain.events import get_event_writer
from omnitrain.jobs import abort, run_core
from omnitrain.jobs.executors import run_in_executor, train_executor
from omnitrain.services.run_evaluation_store import record_evaluation
from omnitrain.services.run_models import ensure_run_custom_model

logger = logging.getLogger("omnitrain.jobs.train")


@dataclass
class RunSpec(run_core.RunSpec):
    cancel_requested: bool


class _WriterSink:
    """`run_core.ProgressSink` for a run training in this process: the event writer and the abort Event."""

    def __init__(self, run_id: str, writer, abort_event) -> None:
        self._run_id = run_id
        self._writer = writer
        self._abort_event = abort_event

    def heartbeat(self) -> None:
        self._writer.touch(self._run_id)

    def check_abort(self) -> None:
        _check_abort(self._run_id, self._abort_event)

    def report(self, epoch: int, split: str, metrics: dict[str, float]) -> None:
        self._writer.metric(self._run_id, epoch, split, metrics)


async def _load_run(run_id: uuid.UUID) -> RunSpec | None:
    async with get_sessionmaker()() as s:
        run = await s.get(TrainingRun, run_id)
        if run is None:
            return None
        return RunSpec(
            run.id, run.backend, run.config or {}, run.dataset_version_id, run.config_key,
            run.cancel_requested_at is not None,
        )  # fmt: skip


async def run_train(run_id: uuid.UUID) -> None:
    """Run a claimed training job to a terminal status. Never raises."""
    writer = get_event_writer()
    rid = str(run_id)
    # Register the abort Event BEFORE reading the DB cancel flag (see abort.py for why the order matters).
    abort_event = abort.register(rid)
    try:
        run = await _load_run(run_id)
        if run is None:
            logger.error("Training run %s vanished before it started", rid)
            return
        if run.cancel_requested:
            abort_event.set()
        if abort_event.is_set():
            logger.info("Run %s was canceled before training started", rid)
            writer.status(rid, "canceled")
            return

        writer.status(rid, "running")
        logger.info("Starting training for run %s", rid)

        # A custom model's files live at the path the compiled config names; make sure they are there.
        await ensure_run_custom_model(run_id)

        sink = _WriterSink(rid, writer, abort_event)
        evaluation = await run_in_executor(train_executor, run_core.execute, run, sink)

        if evaluation is not None:
            await record_evaluation(run_id, evaluation)
        writer.status(rid, "succeeded")
        logger.info("Training completed for run %s", rid)

    except abort.TrainingAborted:
        logger.info("Training aborted for run %s", rid)
        writer.status(rid, "canceled")
    except Exception as e:
        logger.exception("Training failed for run %s", rid)
        writer.status(rid, "failed", f"{type(e).__name__}: {e}")
    finally:
        abort.unregister(rid)


def _check_abort(run_id: str, abort_event) -> None:
    if abort_event.is_set():
        raise abort.TrainingAborted(f"Training aborted for run {run_id}")


async def handle_failure(job_id: uuid.UUID, exc: BaseException) -> None:
    """Backstop for a bug that escaped run_train (which normally never raises). Training is never retried."""
    logger.error("Training job %s escaped its handler: %s", job_id, exc)
    get_event_writer().status(job_id, "failed", f"{type(exc).__name__}: {exc}")
