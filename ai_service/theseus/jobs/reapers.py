"""Periodic housekeeping, one loop, one function per concern (each independently testable).

  stale_runs               a running run with no event for a long time -> failed (hung training thread)
  expired_leases           export jobs whose worker stopped renewing -> re-queued or failed
  event_retention          old log rows and old run events
  rate_limit               expired API-key rate-limit windows

A run marked failed by stale_runs whose training thread is genuinely wedged still occupies the
train lane: a Python thread cannot be killed. The abort Event is set as a best effort, which
stops it at the next epoch boundary if it is merely slow rather than dead.
"""

import asyncio
import logging

import sqlalchemy as sa

from theseus.auth import rate_limit
from theseus.db.base import get_sessionmaker
from theseus.db.models import RunEvent, TrainingRun
from theseus.events import get_event_writer
from theseus.jobs import abort, queue
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

REAP_INTERVAL_SECONDS = 60.0
LOG_RETENTION_AFTER_TERMINAL = "24 hours"
EVENT_RETENTION = "7 days"


async def stale_runs(timeout_seconds: int | None = None) -> int:
    timeout = timeout_seconds if timeout_seconds is not None else get_settings().run_heartbeat_timeout_seconds
    cutoff = sa.func.now() - sa.func.make_interval(0, 0, 0, 0, 0, 0, timeout)
    async with get_sessionmaker()() as s:
        ids = (
            (
                await s.execute(
                    sa.select(TrainingRun.id).where(
                        TrainingRun.status == "running",
                        sa.func.coalesce(TrainingRun.heartbeat_at, TrainingRun.updated_at) < cutoff,
                    )
                )
            )
            .scalars()
            .all()
        )
    writer = get_event_writer()
    for run_id in ids:
        writer.status(run_id, "failed", "Run heartbeat timed out: the training process stopped responding")
        abort.signal(str(run_id))
    return len(ids)


async def expired_leases() -> int:
    ids = await queue.requeue_expired(queue.EXPORT, {"failed_message": "The export worker stopped responding"})
    return len(ids)


async def event_retention() -> int:
    async with get_sessionmaker()() as s:
        logs = await s.execute(
            sa.delete(RunEvent).where(
                RunEvent.kind == "log",
                RunEvent.run_id.in_(
                    sa.select(TrainingRun.id).where(
                        TrainingRun.completed_at < sa.func.now() - sa.text(f"interval '{LOG_RETENTION_AFTER_TERMINAL}'")
                    )
                ),
            )
        )
        old = await s.execute(
            sa.delete(RunEvent).where(RunEvent.ts < sa.func.now() - sa.text(f"interval '{EVENT_RETENTION}'"))
        )
        await s.commit()
    return (logs.rowcount or 0) + (old.rowcount or 0)


async def sweep_rate_limits() -> int:
    return rate_limit.sweep(60)


TASKS = (stale_runs, expired_leases, event_retention, sweep_rate_limits)


async def run_once() -> dict[str, int]:
    """Run every reaper once. One failing task never stops the others."""
    results: dict[str, int] = {}
    for task in TASKS:
        try:
            results[task.__name__] = await task()
        except Exception:
            logger.exception("Reaper %s failed", task.__name__)
    return results


async def reaper_loop(interval: float = REAP_INTERVAL_SECONDS) -> None:
    while True:
        await asyncio.sleep(interval)
        results = await run_once()
        if any(results.values()):
            logger.info("Reaper results: %s", {k: v for k, v in results.items() if v})
