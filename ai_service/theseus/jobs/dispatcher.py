"""Lane loops: claim queued jobs from Postgres and run them in this process.

Three lanes (see executors.py): train (1 at a time, GPU), export (2), inference (N). Each lane
loop does, in this order:

    acquire a lane slot  ->  claim one job  ->  run it  ->  release the slot

Acquire the slot BEFORE claiming. Claiming first and then awaiting a saturated semaphore would
leave a row marked `running` that never ran if the process died in between.

Jobs are never started with BackgroundTasks or a bare create_task: the loop keeps only a weak
reference, so a bare task can be garbage collected mid-flight. Every task is held in a strong
set with a done-callback that discards it and logs any escaped error.
"""

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from theseus import metrics as otel
from theseus.jobs import abort, queue
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

RunFn = Callable[[uuid.UUID], Awaitable[None]]
FailFn = Callable[[uuid.UUID, BaseException], Awaitable[None]]


@dataclass
class Lane:
    name: str
    kind: queue.JobKind
    concurrency: int
    run: RunFn
    on_failure: FailFn | None = None
    # Export and inference jobs hold a lease that must be renewed while they run. Training never
    # uses lease reclaim: a hung thread cannot be taken over, and a restart must not re-train.
    renew_lease: bool = False
    semaphore: asyncio.Semaphore = field(init=False)
    wakeup: asyncio.Event = field(init=False)
    running: set[uuid.UUID] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.wakeup = asyncio.Event()


class Dispatcher:
    def __init__(
        self,
        lanes: list[Lane],
        *,
        poll_interval: float | None = None,
        lease_seconds: int | None = None,
        renew_interval: float | None = None,
        worker_id: str | None = None,
    ) -> None:
        s = get_settings()
        self.lanes = {lane.name: lane for lane in lanes}
        self._poll = poll_interval if poll_interval is not None else s.job_poll_interval_seconds
        self._lease = lease_seconds if lease_seconds is not None else s.job_lease_seconds
        self._renew_interval = renew_interval
        self._worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._stopping = False
        self._loops: list[asyncio.Task] = []
        self._tasks: set[asyncio.Task] = set()

    # -- Lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        self._stopping = False
        for lane in self.lanes.values():
            self._loops.append(asyncio.create_task(self._lane_loop(lane), name=f"lane-{lane.name}"))

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop claiming, ask training threads to stop at their next epoch, wait briefly for jobs."""
        self._stopping = True
        for lane in self.lanes.values():
            lane.wakeup.set()
        for t in self._loops:
            t.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)
        self._loops.clear()
        abort.signal_all()
        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=timeout)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    # -- Public helpers ----------------------------------------------------------------------

    def nudge(self, lane_name: str) -> None:
        """Wake a lane right after enqueueing, so pickup is immediate. The poll is only a fallback."""
        lane = self.lanes.get(lane_name)
        if lane is not None:
            lane.wakeup.set()

    def running_count(self, lane_name: str) -> int:
        return len(self.lanes[lane_name].running)

    def has_capacity(self, lane_name: str) -> bool:
        lane = self.lanes[lane_name]
        return len(lane.running) < lane.concurrency

    # -- Internals ---------------------------------------------------------------------------

    async def _lane_loop(self, lane: Lane) -> None:
        while not self._stopping:
            await lane.semaphore.acquire()
            claimed = False
            try:
                # Clear BEFORE claiming, so an enqueue racing the claim re-sets it.
                lane.wakeup.clear()
                job_id = await queue.claim_one(lane.kind, self._worker_id, self._lease)
                if job_id is not None:
                    claimed = True
                    self._spawn(lane, job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Lane %s failed to claim a job", lane.name)
                job_id = None
            finally:
                if not claimed:
                    lane.semaphore.release()
            if not claimed:
                try:
                    await asyncio.wait_for(lane.wakeup.wait(), self._poll)
                except TimeoutError:
                    pass

    def _spawn(self, lane: Lane, job_id: uuid.UUID) -> None:
        lane.running.add(job_id)
        task = asyncio.create_task(self._run_job(lane, job_id), name=f"{lane.name}-{job_id}")
        self._tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._tasks.discard(t)
            lane.running.discard(job_id)
            lane.semaphore.release()
            if not t.cancelled() and t.exception() is not None:
                logger.error("Job task for %s %s escaped its handler", lane.name, job_id, exc_info=t.exception())

        task.add_done_callback(_done)

    async def _run_job(self, lane: Lane, job_id: uuid.UUID) -> None:
        started = time.monotonic()
        outcome = "success"
        renewer = asyncio.create_task(self._renew_loop(lane, job_id)) if lane.renew_lease else None
        try:
            await lane.run(job_id)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            outcome = "error"
            logger.exception("%s job %s failed", lane.name, job_id)
            if lane.on_failure is not None:
                try:
                    await lane.on_failure(job_id, exc)
                except Exception:
                    logger.exception("Failure handler for %s job %s also failed", lane.name, job_id)
        finally:
            if renewer is not None:
                renewer.cancel()
            attrs = {"lane": lane.name, "outcome": outcome}
            otel.job_duration.record((time.monotonic() - started) * 1000, attrs)
            otel.job_count.add(1, attrs)

    async def _renew_loop(self, lane: Lane, job_id: uuid.UUID) -> None:
        interval = self._renew_interval if self._renew_interval is not None else max(1.0, self._lease / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                await queue.renew_lease(lane.kind, job_id, self._lease)
            except Exception:
                logger.warning("Could not renew lease for %s job %s", lane.name, job_id, exc_info=True)


# -- Process-wide accessor -------------------------------------------------------------------

_dispatcher: Dispatcher | None = None


def set_dispatcher(d: Dispatcher | None) -> None:
    global _dispatcher
    _dispatcher = d


def get_dispatcher() -> Dispatcher | None:
    return _dispatcher


def nudge(lane_name: str) -> None:
    """Wake a lane if a dispatcher is running (a no-op otherwise, e.g. in tests)."""
    if _dispatcher is not None:
        _dispatcher.nudge(lane_name)


def build_default_lanes() -> list[Lane]:
    """The real lanes. Imported lazily because train/export pull in torch and Ludwig."""
    from theseus.jobs import export, inference, train

    s = get_settings()
    return [
        Lane("train", queue.TRAIN, 1, train.run_train, train.handle_failure),
        Lane("export", queue.EXPORT, 2, export.run_export, export.handle_failure, renew_lease=True),
        Lane(
            "inference",
            queue.INFERENCE,
            s.inference_concurrency,
            inference.run_inference,
            inference.handle_failure,
            renew_lease=True,
        ),
    ]
