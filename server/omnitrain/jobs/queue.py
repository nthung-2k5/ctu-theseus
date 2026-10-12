"""The domain tables ARE the queue.

There is no separate jobs table: that would mean writing every transition twice, with a real
dual-truth failure mode (job row failed, run row still running). Instead training_runs,
and exports each carry a few job columns (attempt, max_attempts, available_at,
claimed_by, lease_expires_at, last_error) and are claimed with FOR UPDATE SKIP LOCKED.

House style for every transition here: the status column is the lock. Each change is a guarded
compare-and-swap (UPDATE ... WHERE status = <expected> RETURNING id) and zero rows means
someone else got there first, so do nothing.

Training can run in two places. The server's own `train` lane claims with `claim_one(TRAIN, ...)`, which holds a run
back briefly while an idle remote worker could take it (`_not_deferred_to_worker`). A remote worker claims with
`claim_for_worker`. Both are the same compare-and-swap on the same row, so a run is won by exactly one of them.
"""

import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import aliased

from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import CustomModel, ModelExport, TrainingRun, TrainingWorker
from omnitrain.settings import get_settings

logger = logging.getLogger(__name__)


def _seconds(n: float) -> Any:
    return sa.func.make_interval(0, 0, 0, 0, 0, 0, n)


def remote_custom_backends() -> list[str]:
    """Ids of the installed backends that can train a bring-your-own model on a remote worker
    (`TrainerBackend.remote_custom_models`). Looked up when needed: the registry imports the backends."""
    from omnitrain.backends.registry import list_backends

    return [b.id for b in list_backends() if b.remote_custom_models]


def _worker_can_take_model(custom_backends: Sequence[str]) -> Any:
    """SQL condition: a worker may train this queued run as far as its model goes.

    True for a run on a built-in model. For a bring-your-own model it must be one the worker can fetch by itself,
    because a worker has no database and never gets anyone's Hugging Face token: an uploaded bundle (from S3, with
    its checksum re-checked) or a public Hub model at its pinned revision; a private or gated Hub model stays on the
    server. And the run's backend must have opted in (`custom_backends`): it has to save a trained model that does not
    point back at the model's files, since those live at a different path on the worker.
    """
    run = TrainingRun
    model = aliased(CustomModel)
    fetchable = (
        sa.select(model.id)
        .where(
            model.id == run.custom_model_id,
            sa.or_(
                model.source_kind == "upload",
                sa.and_(
                    model.source_kind == "hub", sa.func.coalesce(model.spec["private"].as_boolean(), False).is_(False)
                ),
            ),
        )
        .exists()
    )
    return sa.or_(run.custom_model_id.is_(None), sa.and_(run.backend.in_(list(custom_backends)), fetchable))


def _not_deferred_to_worker() -> Any:
    """SQL condition: this queued run should be trained by the server rather than wait for a remote worker.

    False (the run is held back) only while ALL of these hold: the run is younger than `remote_claim_grace_seconds`,
    a worker may train its model at all (`_worker_can_take_model`: built-in, or one a worker can fetch), and some
    enabled, non-revoked worker that was seen within `worker_online_seconds`, can train the run's backend and has no run
    in progress. With no workers registered that is never true, so the server claims exactly as it always did.
    """
    s = get_settings()
    run = TrainingRun
    other = aliased(TrainingRun)
    idle_worker = (
        sa.select(TrainingWorker.id)
        .where(
            TrainingWorker.enabled,
            TrainingWorker.revoked_at.is_(None),
            TrainingWorker.last_seen_at > sa.func.now() - _seconds(s.worker_online_seconds),
            TrainingWorker.info["backends"].has_key(run.backend),
            ~sa.exists().where(other.worker_id == TrainingWorker.id, other.status == "running"),
        )
        .exists()
    )
    waiting_for_worker = sa.and_(
        _worker_can_take_model(remote_custom_backends()),
        run.created_at > sa.func.now() - _seconds(s.remote_claim_grace_seconds),
        idle_worker,
    )
    return ~waiting_for_worker


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
    # Extra condition `claim_one` (the server's own claim) adds to the queued-row filter, evaluated per claim.
    claim_guard: Callable[[], Any] | None = None


TRAIN = JobKind("train", TrainingRun, "queued", ("running",), "running", "failed", claim_guard=_not_deferred_to_worker)
EXPORT = JobKind("export", ModelExport, "pending", ("converting", "assembling"), "converting", "failed")
# A bring-your-own model is validated (pinned, fetched, checked) before anyone can train on it.
VALIDATE = JobKind("validate", CustomModel, "uploaded", ("validating",), "validating", "failed")


async def _claim(
    session: Any,
    kind: JobKind,
    claimed_by: str,
    lease_seconds: int,
    where: Sequence[Any] = (),
    values: dict[str, Any] | None = None,
) -> uuid.UUID | None:
    """The claim statement itself: the oldest available queued row matching `where` moves to `kind.claimed`.

    Does not commit. `values` are extra columns set on the claimed row in the same UPDATE.
    """
    m = kind.model
    nxt = (
        sa.select(m.id)
        .where(m.status == kind.queued, m.available_at <= sa.func.now(), *where)
        .order_by(m.created_at, m.id)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    row = (
        await session.execute(
            sa.update(m)
            .where(m.id == nxt)
            .values(
                status=kind.claimed,
                attempt=m.attempt + 1,
                claimed_by=claimed_by,
                lease_expires_at=sa.func.now() + _seconds(lease_seconds),
                **(values or {}),
            )
            .returning(m.id)
        )
    ).first()
    return row[0] if row else None


async def claim_one(kind: JobKind, worker_id: str, lease_seconds: int) -> uuid.UUID | None:
    """Atomically claim the oldest available queued job, or None. This is the server's own claim.

    The claim commits immediately. lease_expires_at is the lease; the FOR UPDATE lock is NOT held
    for the life of the job (a multi-hour open transaction would block vacuum and trip statement
    timeouts).
    """
    where = [kind.claim_guard()] if kind.claim_guard is not None else []
    async with get_sessionmaker()() as session:
        job_id = await _claim(session, kind, worker_id, lease_seconds, where)
        await session.commit()
    return job_id


def worker_claimed_by(worker_id: uuid.UUID) -> str:
    """The `claimed_by` a remote worker's claim records, which tells it apart from a server's `host:pid`."""
    return f"worker:{worker_id}"


async def claim_for_worker(
    worker_id: uuid.UUID,
    backends: Sequence[str],
    lease_seconds: int,
    custom_backends: Sequence[str] | None = None,
) -> uuid.UUID | None:
    """Claim the oldest queued training run for a remote worker, or None.

    Only a run for one of `backends` is eligible, on a built-in model or a bring-your-own one the worker can fetch
    (`_worker_can_take_model`; `custom_backends` defaults to the installed backends that opted in), and a worker holds
    one run at a time. The worker's own row is locked for the duration, so two claims from one worker (a duplicated
    process) are serialised and the second sees the first's run as in progress instead of racing it.
    """
    m = TrainingRun
    async with get_sessionmaker()() as session:
        usable = (
            await session.execute(
                sa.select(TrainingWorker.id)
                .where(TrainingWorker.id == worker_id, TrainingWorker.enabled, TrainingWorker.revoked_at.is_(None))
                .with_for_update()
            )
        ).first()
        if usable is None:
            return None
        busy = (
            await session.execute(sa.select(m.id).where(m.worker_id == worker_id, m.status == "running").limit(1))
        ).first()
        if busy is not None:
            return None
        job_id = await _claim(
            session,
            TRAIN,
            worker_claimed_by(worker_id),
            lease_seconds,
            where=[
                m.backend.in_(list(backends)),
                _worker_can_take_model(remote_custom_backends() if custom_backends is None else custom_backends),
            ],
            values={"worker_id": worker_id},
        )
        await session.commit()
    return job_id


async def renew_lease(kind: JobKind, job_id: uuid.UUID, lease_seconds: int) -> bool:
    m = kind.model
    async with get_sessionmaker()() as session:
        res = await session.execute(
            sa.update(m)
            .where(m.id == job_id, m.status.in_(kind.in_flight))
            .values(lease_expires_at=sa.func.now() + _seconds(lease_seconds))
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
        "available_at": sa.func.now() + _seconds(kind.retry_delay_seconds),
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
