"""The domain tables ARE the queue.

There is no separate jobs table: that would mean writing every transition twice, with a real
dual-truth failure mode (job row failed, run row still running). Instead training_runs,
and exports each carry a few job columns (attempt, max_attempts, available_at,
claimed_by, lease_expires_at, last_error) and are claimed with FOR UPDATE SKIP LOCKED.

House style for every transition here: the status column is the lock. Each change is a guarded
compare-and-swap (UPDATE ... WHERE status = <expected> RETURNING id) and zero rows means
someone else got there first, so do nothing.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa

from theseus.db.base import get_sessionmaker
from theseus.db.models import ModelExport, TrainingRun

logger = logging.getLogger(__name__)


def _status_case(kind: "JobKind", exhausted: Any) -> Any:
    """CASE WHEN attempts exhausted THEN failed ELSE queued, typed as the model status enum.

    Bound as plain strings Postgres would reject the assignment to a native enum column.
    """
    t = kind.model.status.type
    return sa.case((exhausted, sa.literal(kind.failed, t)), else_=sa.literal(kind.queued, t))


@dataclass(frozen=True)
class JobKind:
    name: str
    model: Any
    queued: str  # status while waiting to be claimed
    in_flight: tuple[str, ...]  # statuses while a worker holds the job
    claimed: str  # status a claim moves the row to
    failed: str  # status when attempts are exhausted
    retry_delay_seconds: float = 30.0  # the old nak_delay


TRAIN = JobKind("train", TrainingRun, "queued", ("running",), "running", "failed")
EXPORT = JobKind("export", ModelExport, "pending", ("converting", "assembling"), "converting", "failed")


async def claim_one(kind: JobKind, worker_id: str, lease_seconds: int) -> uuid.UUID | None:
    """Atomically claim the oldest available queued job, or None.

    The claim commits immediately. lease_expires_at is the lease; the FOR UPDATE lock is NOT held
    for the life of the job (a multi-hour open transaction would block vacuum and trip statement
    timeouts).
    """
    m = kind.model
    nxt = (
        sa.select(m.id)
        .where(m.status == kind.queued, m.available_at <= sa.func.now())
        .order_by(m.created_at, m.id)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                sa.update(m)
                .where(m.id == nxt)
                .values(
                    status=kind.claimed,
                    attempt=m.attempt + 1,
                    claimed_by=worker_id,
                    lease_expires_at=sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds),
                )
                .returning(m.id)
            )
        ).first()
        await session.commit()
    return row[0] if row else None


async def renew_lease(kind: JobKind, job_id: uuid.UUID, lease_seconds: int) -> bool:
    m = kind.model
    async with get_sessionmaker()() as session:
        res = await session.execute(
            sa.update(m)
            .where(m.id == job_id, m.status.in_(kind.in_flight))
            .values(lease_expires_at=sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds))
            .returning(m.id)
        )
        found = res.first() is not None
        await session.commit()
    return found


async def release_or_fail(
    kind: JobKind, job_id: uuid.UUID, error: str, final_values: dict[str, Any] | None = None
) -> str | None:
    """A job attempt failed: re-queue after a delay, or fail it if attempts are exhausted.

    Returns the new status (queued or failed), or None if the job was no longer in flight (it
    finished, or was recovered, in the meantime). final_values are extra columns to set only when
    the job fails for good (failed_message, completed_at, ...).
    """
    m = kind.model
    exhausted = m.attempt >= m.max_attempts
    values: dict[str, Any] = {
        "status": _status_case(kind, exhausted),
        "available_at": sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, kind.retry_delay_seconds),
        "claimed_by": None,
        "lease_expires_at": None,
        "last_error": error[:2000],
    }
    for col, val in (final_values or {}).items():
        values[col] = sa.case((exhausted, val), else_=getattr(m, col))
    async with get_sessionmaker()() as session:
        res = await session.execute(
            sa.update(m).where(m.id == job_id, m.status.in_(kind.in_flight)).values(**values).returning(m.status)
        )
        row = res.first()
        await session.commit()
    return row[0] if row else None


async def requeue_expired(kind: JobKind, final_values: dict[str, Any] | None = None) -> list[uuid.UUID]:
    """Re-queue (or fail) in-flight jobs whose lease ran out. Never used for training: a hung
    training thread cannot be reclaimed, and a restarted training run must not silently re-burn
    GPU hours."""
    m = kind.model
    exhausted = m.attempt >= m.max_attempts
    values: dict[str, Any] = {
        "status": _status_case(kind, exhausted),
        "available_at": sa.func.now(),
        "claimed_by": None,
        "lease_expires_at": None,
        "last_error": "Lease expired (the worker stopped renewing it)",
    }
    for col, val in (final_values or {}).items():
        values[col] = sa.case((exhausted, val), else_=getattr(m, col))
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(
                sa.update(m)
                .where(m.status.in_(kind.in_flight), m.lease_expires_at < sa.func.now())
                .values(**values)
                .returning(m.id)
            )
        ).all()
        await session.commit()
    return [r[0] for r in rows]
