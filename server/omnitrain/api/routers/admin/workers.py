"""/api/admin/workers: register, switch off and revoke the remote GPU machines that may train runs.

A worker is registered here, gets a bearer token shown exactly once, and from then on pulls work through /api/worker
(routers/worker.py). The token is hashed like an API key. Revoking is soft: the row stays so runs keep showing where
they trained.
"""

import uuid
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Response

from omnitrain.api.deps import AdminId, SessionDep
from omnitrain.auth.worker_tokens import generate_worker_token
from omnitrain.db.models import TrainingRun, TrainingWorker
from omnitrain.events import get_event_writer
from omnitrain.schemas.admin import CreatedWorker, CreateWorkerBody, UpdateWorkerBody, WorkerListResponse, WorkerOut
from omnitrain.settings import get_settings

router = APIRouter()


def _out(worker: TrainingWorker, current: tuple[uuid.UUID, str] | None) -> WorkerOut:
    info = worker.info or {}
    online_after = datetime.now(UTC) - timedelta(seconds=get_settings().worker_online_seconds)
    return WorkerOut(
        id=worker.id,
        name=worker.name,
        enabled=worker.enabled,
        online=worker.revoked_at is None and worker.last_seen_at is not None and worker.last_seen_at > online_after,
        last_seen_at=worker.last_seen_at,
        created_at=worker.created_at,
        revoked_at=worker.revoked_at,
        hostname=info.get("hostname"),
        gpu=info.get("gpu"),
        version=info.get("version"),
        backends=list(info.get("backends") or []),
        current_run_id=current[0] if current else None,
        current_run_name=current[1] if current else None,
    )


async def _running_runs(session: SessionDep, worker_ids: list[uuid.UUID]) -> dict[uuid.UUID, tuple[uuid.UUID, str]]:
    rows = (
        await session.execute(
            sa.select(TrainingRun.worker_id, TrainingRun.id, TrainingRun.name).where(
                TrainingRun.worker_id.in_(worker_ids), TrainingRun.status == "running"
            )
        )
    ).all()
    return {worker_id: (run_id, name) for worker_id, run_id, name in rows}


@router.get("/workers", response_model=WorkerListResponse)
async def admin_list_workers(session: SessionDep) -> WorkerListResponse:
    workers = (
        (
            await session.execute(
                sa.select(TrainingWorker).order_by(
                    TrainingWorker.revoked_at.is_not(None), TrainingWorker.created_at.desc(), TrainingWorker.id.desc()
                )
            )
        )
        .scalars()
        .all()
    )
    running = await _running_runs(session, [w.id for w in workers])
    return WorkerListResponse(workers=[_out(w, running.get(w.id)) for w in workers])


@router.post("/workers", status_code=201, response_model=CreatedWorker)
async def admin_create_worker(body: CreateWorkerBody, admin_id: AdminId, session: SessionDep) -> CreatedWorker:
    raw, token_hash = generate_worker_token()
    worker = TrainingWorker(name=body.name.strip(), token_hash=token_hash, created_by=admin_id, info={})
    session.add(worker)
    await session.commit()
    await session.refresh(worker)
    # The only moment the raw token exists: it cannot be derived from the hash, so losing it means registering again.
    return CreatedWorker(worker=_out(worker, None), token=raw)


@router.patch("/workers/{worker_id}", response_model=WorkerOut)
async def admin_update_worker(worker_id: uuid.UUID, body: UpdateWorkerBody, session: SessionDep) -> WorkerOut:
    """Switch a worker on or off. Off only stops it taking NEW runs: one it is training carries on."""
    worker = (
        await session.execute(
            sa.update(TrainingWorker)
            .where(TrainingWorker.id == worker_id, TrainingWorker.revoked_at.is_(None))
            .values(enabled=body.enabled)
            .returning(TrainingWorker)
        )
    ).scalar_one_or_none()
    if worker is None:
        raise HTTPException(404, "Worker not found")
    await session.commit()
    return _out(worker, (await _running_runs(session, [worker.id])).get(worker.id))


@router.delete("/workers/{worker_id}", status_code=204)
async def admin_revoke_worker(worker_id: uuid.UUID, session: SessionDep) -> Response:
    """Revoke a worker's token for good. Its in-progress run is failed at once rather than left to the heartbeat
    reaper: a revoked worker can no longer report anything, so nothing would ever finish it."""
    revoked = (
        await session.execute(
            sa.update(TrainingWorker)
            .where(TrainingWorker.id == worker_id, TrainingWorker.revoked_at.is_(None))
            .values(revoked_at=sa.func.now(), enabled=False)
            .returning(TrainingWorker.id)
        )
    ).first()
    if revoked is None:
        raise HTTPException(404, "Worker not found")
    stranded = (
        (
            await session.execute(
                sa.select(TrainingRun.id).where(TrainingRun.worker_id == worker_id, TrainingRun.status == "running")
            )
        )
        .scalars()
        .all()
    )
    await session.commit()
    for run_id in stranded:
        get_event_writer().status(run_id, "failed", "The worker was revoked while training this run")
    return Response(status_code=204)
