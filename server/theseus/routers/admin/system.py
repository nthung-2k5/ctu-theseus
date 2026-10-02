"""/api/admin/system: a one-screen picture of the platform: how much there is, how busy each job lane is,
and whether each trainer backend can actually run.

Everything is read from Postgres or from the in-process dispatcher, never from object storage: a bucket
listing grows with the data and would make this page slower the more it is used.
"""

import sqlalchemy as sa
from fastapi import APIRouter

from theseus.backends.registry import list_backends
from theseus.db.models import CustomModel, ModelExport, Project, TrainingRun, User
from theseus.deps import SessionDep
from theseus.jobs.dispatcher import get_dispatcher
from theseus.schemas.admin import BackendStatusOut, LaneOut, SystemCounts, SystemResponse

router = APIRouter()

# The lanes, with what "waiting" means for each (a row in the status the dispatcher claims from).
_LANES = (
    ("train", "Training", TrainingRun, "queued"),
    ("export", "Export", ModelExport, "pending"),
    ("validate", "Model validation", CustomModel, "uploaded"),
)


async def _count(session: SessionDep, model, *where) -> int:
    return (await session.execute(sa.select(sa.func.count()).select_from(model).where(*where))).scalar_one()


@router.get("/system", response_model=SystemResponse)
async def admin_system(session: SessionDep) -> SystemResponse:
    model_bytes = (
        await session.execute(
            sa.select(sa.func.coalesce(sa.func.sum(CustomModel.size_bytes), 0)).where(CustomModel.status == "ready")
        )
    ).scalar_one()
    counts = SystemCounts(
        users=await _count(session, User),
        admins=await _count(session, User, User.role == "admin", User.disabled_at.is_(None)),
        disabled_users=await _count(session, User, User.disabled_at.is_not(None)),
        projects=await _count(session, Project),
        runs=await _count(session, TrainingRun),
        active_runs=await _count(session, TrainingRun, TrainingRun.status.in_(("queued", "running"))),
        exports=await _count(session, ModelExport),
        custom_models=await _count(session, CustomModel, CustomModel.archived_at.is_(None)),
        custom_model_bytes=int(model_bytes),
    )

    dispatcher = get_dispatcher()
    lanes = []
    for name, label, model, waiting in _LANES:
        lane = dispatcher.lanes.get(name) if dispatcher is not None else None
        lanes.append(
            LaneOut(
                name=name,
                label=label,
                queued=await _count(session, model, model.status == waiting),
                running=len(lane.running) if lane is not None else None,
                capacity=lane.concurrency if lane is not None else None,
            )
        )

    backends = [
        BackendStatusOut(id=b.id, label=b.label, available=(reason := b.available()) is None, unavailable_reason=reason)
        for b in list_backends()
    ]
    return SystemResponse(counts=counts, lanes=lanes, backends=backends)
