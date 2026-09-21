"""The single writer of run_events, and the place run state changes are applied.

Two invariants make this module load-bearing:

1. Nothing else inserts into run_events. seq is a BIGSERIAL-style identity, which is NOT
   monotonic in commit order when several transactions insert concurrently (seq 6 can commit
   before seq 5). A reader that emits `id: 6` would then never see 5 on reconnect. With exactly
   one writer, seq order equals commit order, which is what Last-Event-ID replay relies on.

2. Every run state change goes through here as a guarded compare-and-swap
   (UPDATE ... WHERE status NOT IN terminal RETURNING id). Zero rows means the run already
   finished, or someone else got there first, so the change is dropped and NO event is
   recorded. That is what stops a late epoch event from resurrecting a run the cancel endpoint
   or the reaper already ended.

Each event's projection (training_runs status, training_metrics, best_epoch) is applied in the
same transaction as its run_events row, so the event log and the tables cannot diverge.

Emitters (the training thread, request handlers, reapers) call the thread-safe methods below,
which enqueue onto one asyncio.Queue drained by one task.
"""

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import metrics as otel
from theseus.db.base import get_sessionmaker
from theseus.db.enums import TERMINAL_RUN_STATUSES
from theseus.db.models import RunEvent, TrainingMetric, TrainingRun
from theseus.events.bus import RunEventBus
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

HEARTBEAT_MIN_INTERVAL_SECONDS = 15.0
BATCH_MAX = 200
# Log lines are coalesced into one transaction for up to this long.
BATCH_WINDOW_SECONDS = 0.05


@dataclass
class _Event:
    run_id: str
    kind: str
    body: dict[str, Any]


@dataclass
class _Touch:
    run_id: str


@dataclass
class _Flush:
    fut: asyncio.Future


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    """Drop NaN/inf: they are invalid JSON for the payload column and meaningless as chart points."""
    return {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}


class EventWriter:
    def __init__(self, bus: RunEventBus, *, log_cap: int | None = None) -> None:
        self._bus = bus
        self._log_cap = log_cap if log_cap is not None else get_settings().run_log_max_rows
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_Event | _Touch | _Flush] | None = None
        self._task: asyncio.Task | None = None
        self._log_counts: dict[str, int] = {}
        self._last_heartbeat: dict[str, float] = {}

    # -- Lifecycle ---------------------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="run-event-writer")

    async def stop(self) -> None:
        """Drain everything already enqueued, then stop."""
        if self._task is None:
            return
        await self.flush()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def flush(self) -> None:
        """Resolve once every event enqueued before this call has been committed and published."""
        assert self._loop is not None
        fut = self._loop.create_future()
        self._put(_Flush(fut))
        await fut

    # -- Emit API (thread-safe: callable from the training thread or the event loop) ---------

    def status(self, run_id: uuid.UUID | str, status: str, message: str | None = None) -> None:
        body: dict[str, Any] = {"status": status}
        if message:
            body["message"] = message
        self._put(_Event(str(run_id), "status", body))

    def metric(self, run_id: uuid.UUID | str, epoch: int, split: str, metrics: dict[str, float]) -> None:
        self._put(_Event(str(run_id), "metric", {"epoch": epoch, "split": split, "metrics": metrics}))

    def log(self, run_id: uuid.UUID | str, level: str, line: str) -> None:
        self._put(_Event(str(run_id), "log", {"level": level, "line": line}))

    def touch(self, run_id: uuid.UUID | str) -> None:
        """Bump the run heartbeat without recording an event."""
        self._put(_Touch(str(run_id)))

    def _put(self, item: _Event | _Touch | _Flush) -> None:
        if self._loop is None or self._queue is None:
            raise RuntimeError("EventWriter has not been started")
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            # The loop is closed (shutdown). Late events from a dying training thread are dropped.
            logger.debug("Dropping run event: event loop is closed")

    # -- Drain loop --------------------------------------------------------------------------

    async def _run(self) -> None:
        assert self._queue is not None and self._loop is not None
        loop = self._loop
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + BATCH_WINDOW_SECONDS
            while len(batch) < BATCH_MAX and not isinstance(batch[-1], _Flush):
                try:
                    nxt = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._queue.get(), remaining)
                    except TimeoutError:
                        break
                batch.append(nxt)
            try:
                await self._process(batch)
            except Exception:
                logger.exception("Run event writer failed to process a batch")
                for item in batch:
                    if isinstance(item, _Flush) and not item.fut.done():
                        item.fut.set_result(None)

    async def _process(self, batch: list[_Event | _Touch | _Flush]) -> None:
        items = [b for b in batch if not isinstance(b, _Flush)]
        published: list[tuple[str, dict[str, Any]]] = []
        try:
            published = await self._commit(items)
        except Exception:
            logger.exception("Run event batch failed; retrying event by event")
            for item in items:
                try:
                    published += await self._commit([item])
                except Exception:
                    logger.exception("Dropping run event that could not be written: %r", item)

        # Publish only after commit, in seq order.
        for run_id, event in published:
            self._bus.publish(run_id, event)
            if event["payload"]["kind"] == "status" and event["payload"]["status"] in TERMINAL_RUN_STATUSES:
                otel.training_run_terminal_count.add(1, {"status": event["payload"]["status"]})
                self._log_counts.pop(run_id, None)
                self._last_heartbeat.pop(run_id, None)

        for item in batch:
            if isinstance(item, _Flush) and not item.fut.done():
                item.fut.set_result(None)

    async def _commit(self, items: list[_Event | _Touch]) -> list[tuple[str, dict[str, Any]]]:
        published: list[tuple[str, dict[str, Any]]] = []
        async with get_sessionmaker()() as session:
            for item in items:
                if isinstance(item, _Touch):
                    await self._maybe_heartbeat(session, item.run_id)
                else:
                    published.extend(await self._apply(session, item))
            await session.commit()
        return published

    # -- Projection --------------------------------------------------------------------------

    async def _apply(self, s: AsyncSession, ev: _Event) -> list[tuple[str, dict[str, Any]]]:
        now = datetime.now(UTC)
        rid = uuid.UUID(ev.run_id)
        not_terminal = (TrainingRun.id == rid, TrainingRun.status.not_in(TERMINAL_RUN_STATUSES))

        if ev.kind == "status":
            status = ev.body["status"]
            values: dict[str, Any] = {"status": status, "heartbeat_at": now}
            if status == "running":
                values["started_at"] = sa.func.coalesce(TrainingRun.started_at, now)
            if status in TERMINAL_RUN_STATUSES:
                values["completed_at"] = now
                if status == "failed" and ev.body.get("message"):
                    values["failed_message"] = ev.body["message"]
            res = await s.execute(
                sa.update(TrainingRun).where(*not_terminal).values(**values).returning(TrainingRun.id)
            )
            if res.first() is None:
                return []  # already terminal: a stale change, dropped without an event
            payload = {"kind": "status", "runId": ev.run_id, "ts": now.isoformat(), **ev.body}
            return [await self._insert(s, ev.run_id, "status", now, payload)]

        if ev.kind == "metric":
            metrics = _finite(ev.body["metrics"])
            epoch, split = ev.body["epoch"], ev.body["split"]
            res = await s.execute(
                sa.update(TrainingRun)
                .where(*not_terminal)
                .values(status="running", heartbeat_at=now)
                .returning(TrainingRun.best_epoch)
            )
            row = res.first()
            if row is None or not metrics:
                return []
            best_epoch = row[0]
            for name, value in metrics.items():
                stmt = pg_insert(TrainingMetric).values(
                    training_run_id=rid, epoch=epoch, split=split, metric_name=name, metric_value=value
                )
                await s.execute(
                    stmt.on_conflict_do_update(
                        index_elements=["training_run_id", "epoch", "split", "metric_name"],
                        set_={"metric_value": value},
                    )
                )
            # Ludwig always minimizes loss whatever the task, so lowest validation loss so far is a
            # task-agnostic definition of the best epoch.
            if split == "validation" and "loss" in metrics:
                is_best = True
                if best_epoch is not None:
                    previous = (
                        await s.execute(
                            sa.select(TrainingMetric.metric_value).where(
                                TrainingMetric.training_run_id == rid,
                                TrainingMetric.epoch == best_epoch,
                                TrainingMetric.split == "validation",
                                TrainingMetric.metric_name == "loss",
                            )
                        )
                    ).scalar_one_or_none()
                    is_best = previous is None or metrics["loss"] < previous
                if is_best:
                    await s.execute(sa.update(TrainingRun).where(TrainingRun.id == rid).values(best_epoch=epoch))
            payload = {
                "kind": "metric",
                "runId": ev.run_id,
                "ts": now.isoformat(),
                "epoch": epoch,
                "split": split,
                "metrics": metrics,
            }
            return [await self._insert(s, ev.run_id, "metric", now, payload)]

        if ev.kind == "log":
            count = self._log_counts.get(ev.run_id, 0) + 1
            self._log_counts[ev.run_id] = count
            await self._maybe_heartbeat(s, ev.run_id)
            if count > self._log_cap + 1:
                return []
            if count == self._log_cap + 1:
                line = (
                    f"Live log stream truncated after {self._log_cap} lines. "
                    "The complete log is still written to the run logs download."
                )
                ev = _Event(ev.run_id, "log", {"level": "warn", "line": line})
            payload = {"kind": "log", "runId": ev.run_id, "ts": now.isoformat(), **ev.body}
            return [await self._insert(s, ev.run_id, "log", now, payload)]

        raise ValueError(f"Unknown run event kind: {ev.kind}")

    async def _maybe_heartbeat(self, s: AsyncSession, run_id: str) -> None:
        """Throttled: a chatty log stream must not become 10 UPDATEs/sec on one training_runs row."""
        mono = time.monotonic()
        if mono - self._last_heartbeat.get(run_id, -1e9) < HEARTBEAT_MIN_INTERVAL_SECONDS:
            return
        self._last_heartbeat[run_id] = mono
        await s.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == uuid.UUID(run_id), TrainingRun.status.not_in(TERMINAL_RUN_STATUSES))
            .values(heartbeat_at=sa.func.now())
        )

    async def _insert(
        self, s: AsyncSession, run_id: str, kind: str, ts: datetime, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        seq = (
            await s.execute(
                sa.insert(RunEvent)
                .values(run_id=uuid.UUID(run_id), kind=kind, ts=ts, payload=payload)
                .returning(RunEvent.seq)
            )
        ).scalar_one()
        return run_id, {"seq": seq, "runId": run_id, "kind": kind, "payload": payload}
