"""/api/admin/runs and /api/admin/exports: every user's training runs and exports, and the two things an
operator needs to do about them: stop a runaway run, and give a failed export another go.

Both go through the same guarded transitions the owner's own routes use (see routers/training.py and
jobs/queue.py), so an admin acting on a row that has just moved on gets a 409 rather than corrupting it.
"""

import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query, Response

from theseus.db.models import ModelExport, Project, TrainingRun, User
from theseus.deps import SessionDep
from theseus.jobs import abort
from theseus.jobs.dispatcher import nudge
from theseus.schemas.admin import AdminExportListResponse, AdminExportOut, AdminRunListResponse, AdminRunOut

router = APIRouter()

_RUN_STATUS = "^(queued|running|succeeded|failed|canceled)$"
_EXPORT_STATUS = "^(pending|converting|assembling|ready|failed)$"


def _like(text: str) -> str:
    needle = text.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{needle}%"


@router.get("/runs", response_model=AdminRunListResponse)
async def admin_list_runs(
    session: SessionDep,
    status: Annotated[str | None, Query(pattern=_RUN_STATUS)] = None,
    user: Annotated[str | None, Query(max_length=100, description="Part of the owner's email")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> AdminRunListResponse:
    base = (
        sa.select(TrainingRun, Project.name, Project.task, User.email)
        .join(Project, Project.id == TrainingRun.project_id)
        .join(User, User.id == Project.user_id)
    )
    if status:
        base = base.where(TrainingRun.status == status)
    if user:
        base = base.where(sa.func.lower(User.email).like(_like(user), escape="\\"))

    total = (await session.execute(sa.select(sa.func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(
            base.order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
    ).all()
    return AdminRunListResponse(
        runs=[
            AdminRunOut(
                id=run.id,
                name=run.name,
                status=run.status,
                project_id=run.project_id,
                project_name=project_name,
                task=task,
                owner_email=email,
                backend=run.backend,
                created_at=run.created_at,
                started_at=run.started_at,
                completed_at=run.completed_at,
                failed_message=run.failed_message,
            )
            for run, project_name, task, email in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/runs/{run_id}/cancel", status_code=204)
async def admin_cancel_run(run_id: uuid.UUID, session: SessionDep) -> Response:
    """Stop someone's queued or running training run, the same way its owner would."""
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    if run.status not in ("queued", "running"):
        raise HTTPException(409, f"Cannot cancel a run with status '{run.status}': it has already finished")
    if not await abort.request_cancel(run.id):
        raise HTTPException(409, "Cannot cancel a run that has already finished")
    return Response(status_code=204)


@router.get("/exports", response_model=AdminExportListResponse)
async def admin_list_exports(
    session: SessionDep,
    status: Annotated[str | None, Query(pattern=_EXPORT_STATUS)] = None,
    user: Annotated[str | None, Query(max_length=100, description="Part of the owner's email")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> AdminExportListResponse:
    base = (
        sa.select(ModelExport, TrainingRun.name, User.email)
        .join(TrainingRun, TrainingRun.id == ModelExport.run_id)
        .join(User, User.id == ModelExport.user_id)
    )
    if status:
        base = base.where(ModelExport.status == status)
    if user:
        base = base.where(sa.func.lower(User.email).like(_like(user), escape="\\"))

    total = (await session.execute(sa.select(sa.func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        await session.execute(
            base.order_by(ModelExport.created_at.desc(), ModelExport.id.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
    ).all()
    return AdminExportListResponse(
        exports=[
            AdminExportOut(
                id=e.id,
                run_id=e.run_id,
                run_name=run_name,
                format=e.format,
                status=e.status,
                owner_email=email,
                attempt=e.attempt,
                max_attempts=e.max_attempts,
                failed_message=e.failed_message,
                last_error=e.last_error,
                created_at=e.created_at,
                ready_at=e.ready_at,
            )
            for e, run_name, email in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/exports/{export_id}/requeue", status_code=204)
async def admin_requeue_export(export_id: uuid.UUID, session: SessionDep) -> Response:
    """Give a failed export a fresh set of attempts. Only a `failed` export can be requeued: the status
    column is the lock, so this is a compare-and-swap and a row that just moved on is left alone."""
    claimed = await session.execute(
        sa.update(ModelExport)
        .where(ModelExport.id == export_id, ModelExport.status == "failed")
        .values(
            status="pending",
            attempt=0,
            available_at=sa.func.now(),
            claimed_by=None,
            lease_expires_at=None,
            failed_message=None,
            last_error=None,
        )
        .returning(ModelExport.id)
    )
    if claimed.first() is None:
        if await session.get(ModelExport, export_id) is None:
            raise HTTPException(404, "Export not found")
        raise HTTPException(409, "Only a failed export can be requeued")
    await session.commit()
    nudge("export")
    return Response(status_code=204)
