"""FastAPI dependencies: the equivalent of the Elysia .macro() guards in server/routes/auth.ts.

Two auth surfaces stay distinct, as before:
  * current_user_id  - browser session (JWT access cookie), for everything under /api
  * ApiKeyUserId     - bearer API key, for the hosted prediction API under /api/v1

Ownership dependencies all follow one pattern: load the entity together with its project ->
404 if missing -> 403 if the project belongs to someone else -> hand the entity to the handler.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlparse

import sqlalchemy as sa
from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.auth import rate_limit
from theseus.auth.api_keys import hash_api_key
from theseus.auth.jwt import InvalidToken, decode_access_token
from theseus.db.base import get_session, get_sessionmaker
from theseus.db.models import (
    Annotation,
    ApiKey,
    DatasetItem,
    DatasetVersion,
    ModelExport,
    Project,
    Sweep,
    TrainingRun,
)
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"

API_KEY_RATE_LIMIT_MAX = 60
API_KEY_RATE_LIMIT_WINDOW_SECONDS = 60

SessionDep = Annotated[AsyncSession, Depends(get_session)]


# -- Browser session -------------------------------------------------------------------------


async def current_user_id(request: Request) -> uuid.UUID:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        return decode_access_token(token)
    except InvalidToken:
        raise HTTPException(401, "Invalid or expired session") from None


UserId = Annotated[uuid.UUID, Depends(current_user_id)]

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


async def verify_origin(request: Request) -> None:
    """CSRF defence for cookie auth: SameSite=Lax plus an Origin check on unsafe methods.

    A content-type check would not be enough: uploads are multipart/form-data, a CORS-simple
    type that a hostile page can submit cross-site. Bearer-token routes are exempt because a
    browser never attaches those credentials on its own.
    """
    if request.method not in _UNSAFE_METHODS or request.url.path.startswith("/api/v1/"):
        return
    origin = request.headers.get("origin")
    if not origin:
        return  # Browsers always send Origin on cross-site unsafe requests, so no Origin means non-browser.
    host = urlparse(origin).netloc
    allowed_hosts = {request.headers.get("x-forwarded-host"), request.headers.get("host")}
    if host in allowed_hosts or origin.rstrip("/") in get_settings().allowed_origin_list:
        return
    raise HTTPException(403, "Cross-origin request blocked")


# -- Ownership -------------------------------------------------------------------------------


async def _owned(session: AsyncSession, user_id: uuid.UUID, stmt: sa.Select, not_found: str):
    row = (await session.execute(stmt)).first()
    if row is None:
        raise HTTPException(404, not_found)
    entity, project = row[0], row[-1]
    if project.user_id != user_id:
        raise HTTPException(403, "Unauthorized")
    return entity, project


async def owned_project(project_id: uuid.UUID, user_id: UserId, session: SessionDep) -> Project:
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if project.user_id != user_id:
        raise HTTPException(403, "Unauthorized")
    return project


async def owned_version(version_id: uuid.UUID, user_id: UserId, session: SessionDep) -> DatasetVersion:
    stmt = (
        sa.select(DatasetVersion, Project)
        .join(Project, Project.id == DatasetVersion.dataset_id)
        .where(DatasetVersion.id == version_id)
    )
    return (await _owned(session, user_id, stmt, "Version not found"))[0]


async def owned_sweep(sweep_id: uuid.UUID, user_id: UserId, session: SessionDep) -> Sweep:
    stmt = sa.select(Sweep, Project).join(Project, Project.id == Sweep.project_id).where(Sweep.id == sweep_id)
    return (await _owned(session, user_id, stmt, "Sweep not found"))[0]


def _run_stmt(run_id: uuid.UUID) -> sa.Select:
    return (
        sa.select(TrainingRun, Project)
        .join(Project, Project.id == TrainingRun.project_id)
        .where(TrainingRun.id == run_id)
    )


async def owned_run(run_id: uuid.UUID, user_id: UserId, session: SessionDep) -> TrainingRun:
    return (await _owned(session, user_id, _run_stmt(run_id), "Training run not found"))[0]


async def owned_item(item_id: uuid.UUID, user_id: UserId, session: SessionDep) -> DatasetItem:
    stmt = (
        sa.select(DatasetItem, Project)
        .join(Project, Project.id == DatasetItem.dataset_id)
        .where(DatasetItem.id == item_id)
    )
    return (await _owned(session, user_id, stmt, "Item not found"))[0]


async def owned_annotation(annotation_id: uuid.UUID, user_id: UserId, session: SessionDep) -> Annotation:
    stmt = (
        sa.select(Annotation, Project)
        .join(DatasetItem, DatasetItem.id == Annotation.item_id)
        .join(Project, Project.id == DatasetItem.dataset_id)
        .where(Annotation.id == annotation_id)
    )
    return (await _owned(session, user_id, stmt, "Annotation not found"))[0]


async def owned_export(export_id: uuid.UUID, user_id: UserId, session: SessionDep) -> ModelExport:
    stmt = (
        sa.select(ModelExport, Project)
        .join(TrainingRun, TrainingRun.id == ModelExport.run_id)
        .join(Project, Project.id == TrainingRun.project_id)
        .where(ModelExport.id == export_id)
    )
    return (await _owned(session, user_id, stmt, "Export not found"))[0]


@dataclass
class DraftContext:
    draft: DatasetVersion
    project: Project


async def owned_draft(project_id: uuid.UUID, user_id: UserId, session: SessionDep) -> DraftContext:
    """The mutable draft version of a project, together with the project, in one query."""
    stmt = (
        sa.select(DatasetVersion, Project)
        .join(Project, Project.id == DatasetVersion.dataset_id)
        .where(DatasetVersion.dataset_id == project_id, DatasetVersion.version_tag.is_(None))
    )
    draft, project = await _owned(session, user_id, stmt, "Draft dataset not found")
    return DraftContext(draft, project)


ProjectDep = Annotated[Project, Depends(owned_project)]
VersionDep = Annotated[DatasetVersion, Depends(owned_version)]
SweepDep = Annotated[Sweep, Depends(owned_sweep)]
RunDep = Annotated[TrainingRun, Depends(owned_run)]
ItemDep = Annotated[DatasetItem, Depends(owned_item)]
AnnotationDep = Annotated[Annotation, Depends(owned_annotation)]
ExportDep = Annotated[ModelExport, Depends(owned_export)]
DraftDep = Annotated[DraftContext, Depends(owned_draft)]


# -- API keys (hosted prediction API) --------------------------------------------------------

_background: set[asyncio.Task] = set()


async def _touch_key(key_id: uuid.UUID) -> None:
    """Best-effort last_used_at bump on its own session; a failure must never block the request."""
    try:
        async with get_sessionmaker()() as s:
            await s.execute(sa.update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=sa.func.now()))
            await s.commit()
    except Exception:
        logger.debug("Could not update last_used_at for api key %s", key_id, exc_info=True)


async def current_api_key_user(request: Request, response: Response, session: SessionDep) -> uuid.UUID:
    header = request.headers.get("authorization", "")
    raw = header[len("Bearer ") :] if header.startswith("Bearer ") else None
    if not raw:
        raise HTTPException(401, "Missing API key: send it as an Authorization Bearer header")

    key_hash = hash_api_key(raw)
    # Throttle before touching Postgres, keyed by the hash (never the raw secret), whether or
    # not the key turns out to be valid.
    rl = rate_limit.check(key_hash, API_KEY_RATE_LIMIT_MAX, API_KEY_RATE_LIMIT_WINDOW_SECONDS)
    if not rl.allowed:
        raise HTTPException(
            429,
            "Rate limit exceeded: try again shortly",
            headers={
                "Retry-After": str(rl.retry_after),
                "X-RateLimit-Limit": str(rl.limit),
                "X-RateLimit-Remaining": "0",
            },
        )
    response.headers["X-RateLimit-Limit"] = str(rl.limit)
    response.headers["X-RateLimit-Remaining"] = str(rl.remaining)

    key = (
        await session.execute(sa.select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None)))
    ).scalar_one_or_none()
    if key is None:
        raise HTTPException(401, "Invalid or revoked API key")

    task = asyncio.create_task(_touch_key(key.id))
    _background.add(task)
    task.add_done_callback(_background.discard)
    return key.user_id


ApiKeyUserId = Annotated[uuid.UUID, Depends(current_api_key_user)]


async def api_key_run(run_id: uuid.UUID, user_id: ApiKeyUserId, session: SessionDep) -> TrainingRun:
    return (await _owned(session, user_id, _run_stmt(run_id), "Training run not found"))[0]


ApiKeyRunDep = Annotated[TrainingRun, Depends(api_key_run)]
