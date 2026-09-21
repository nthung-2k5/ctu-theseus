"""Cancel a training run.

The database is the truth; a threading.Event is only the fast read path:

  * training_runs.cancel_requested_at is the durable intent (it survives a restart).
  * AbortRegistry holds one threading.Event per run currently training in THIS process. Ludwig
    on_epoch_end runs in the training thread and just checks Event.is_set(): pure memory, no DB
    round trip and no hop back onto the event loop (the legacy worker paid a 5 s timeout hop per
    epoch for this).

Ordering makes the two race-free. request_cancel writes the DB flag FIRST, then sets the Event
if the run is registered. A job registers its Event FIRST, then reads the DB flag. Whichever
order they interleave, the job sees the cancel.

Abort is checked only at epoch boundaries: Ludwig exposes no finer hook, so one very long epoch
cannot be interrupted mid-way.
"""

import logging
import threading
import uuid

import sqlalchemy as sa

from theseus.db.base import get_sessionmaker
from theseus.db.models import TrainingRun
from theseus.events import get_event_writer

logger = logging.getLogger(__name__)


class TrainingAborted(Exception):
    """Raised from a Ludwig callback to stop training.

    Deliberately an Exception, NOT KeyboardInterrupt. KeyboardInterrupt is a BaseException that
    uvicorn treats as a shutdown signal, so raising it inside the API process could take the
    whole HTTP server down with a cancelled run. Ludwig has no bare `except Exception` around
    callbacks (only a `finally` around training), so a plain Exception propagates out of
    model.train() cleanly.
    """


_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def register(run_id: str) -> threading.Event:
    with _lock:
        return _events.setdefault(run_id, threading.Event())


def unregister(run_id: str) -> None:
    with _lock:
        _events.pop(run_id, None)


def is_registered(run_id: str) -> bool:
    return run_id in _events


def signal_all() -> int:
    """Ask every in-flight training thread to stop at its next epoch boundary (used on shutdown)."""
    with _lock:
        for ev in _events.values():
            ev.set()
        return len(_events)


def signal(run_id: str) -> bool:
    ev = _events.get(run_id)
    if ev is not None:
        ev.set()
    return ev is not None


async def request_cancel(run_id: uuid.UUID) -> bool:
    """Cancel a queued or running run. Returns False if it was already finished.

    Emits the terminal `canceled` status immediately, so the UI reflects the cancel at once
    rather than at the next epoch boundary. The training thread stops when it next checks the
    Event, and its own later status events are dropped by the writer terminal guard.
    """
    async with get_sessionmaker()() as session:
        res = await session.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == run_id, TrainingRun.status.in_(("queued", "running")))
            .values(cancel_requested_at=sa.func.coalesce(TrainingRun.cancel_requested_at, sa.func.now()))
            .returning(TrainingRun.id)
        )
        found = res.first() is not None
        await session.commit()
    if not found:
        return False
    signal(str(run_id))
    get_event_writer().status(run_id, "canceled")
    return True
