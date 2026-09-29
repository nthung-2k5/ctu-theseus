"""/api/admin/users: list accounts, change role, disable, and revoke sessions and API keys."""

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query, Response

from theseus.auth import refresh as refresh_tokens
from theseus.db.models import ApiKey, Project, TrainingRun, User
from theseus.deps import AdminId, SessionDep
from theseus.schemas.admin import AdminUserListResponse, AdminUserOut, AdminUserResponse, UpdateUserBody
from theseus.schemas.api_keys import ApiKeyListResponse, ApiKeyOut

router = APIRouter(prefix="/users")


def _count_columns() -> tuple[sa.ScalarSelect, sa.ScalarSelect, sa.ScalarSelect]:
    projects = sa.select(sa.func.count()).select_from(Project).where(Project.user_id == User.id).scalar_subquery()
    runs = (
        sa.select(sa.func.count())
        .select_from(TrainingRun)
        .join(Project, Project.id == TrainingRun.project_id)
        .where(Project.user_id == User.id)
        .scalar_subquery()
    )
    keys = (
        sa.select(sa.func.count())
        .select_from(ApiKey)
        .where(ApiKey.user_id == User.id, ApiKey.revoked_at.is_(None))
        .scalar_subquery()
    )
    return projects, runs, keys


def _out(user: User, projects: int, runs: int, keys: int) -> AdminUserOut:
    return AdminUserOut(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,
        disabled=user.disabled_at is not None,
        created_at=user.created_at,
        project_count=projects,
        run_count=runs,
        api_key_count=keys,
    )


async def _load(session: SessionDep, user_id: uuid.UUID) -> AdminUserOut:
    projects, runs, keys = _count_columns()
    row = (await session.execute(sa.select(User, projects, runs, keys).where(User.id == user_id))).first()
    if row is None:
        raise HTTPException(404, "User not found")
    return _out(*row)


async def _other_active_admins(session: SessionDep, excluding: uuid.UUID) -> int:
    return (
        await session.execute(
            sa.select(sa.func.count()).where(User.role == "admin", User.disabled_at.is_(None), User.id != excluding)
        )
    ).scalar_one()


@router.get("", response_model=AdminUserListResponse)
async def admin_list_users(
    session: SessionDep,
    q: str | None = Query(default=None, max_length=100),
    role: str | None = Query(default=None, pattern="^(user|admin)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
) -> AdminUserListResponse:
    filters = []
    if q:
        needle = q.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{needle}%"
        filters.append(
            sa.or_(
                sa.func.lower(User.email).like(like, escape="\\"),
                sa.func.lower(User.name).like(like, escape="\\"),
            )
        )
    if role:
        filters.append(User.role == role)

    total = (await session.execute(sa.select(sa.func.count()).select_from(User).where(*filters))).scalar_one()
    projects, runs, keys = _count_columns()
    rows = (
        await session.execute(
            sa.select(User, projects, runs, keys)
            .where(*filters)
            .order_by(User.created_at.desc(), User.id)
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
    ).all()
    return AdminUserListResponse(users=[_out(*r) for r in rows], total=total, page=page, page_size=page_size)


@router.patch("/{user_id}", response_model=AdminUserResponse)
async def admin_update_user(
    user_id: uuid.UUID, body: UpdateUserBody, admin_id: AdminId, session: SessionDep
) -> AdminUserResponse:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")

    demoting = body.role == "user" and user.role == "admin"
    disabling = body.disabled is True and user.disabled_at is None
    if (demoting or disabling) and user.id == admin_id:
        raise HTTPException(400, "You cannot demote or disable your own account")
    if (demoting or disabling) and user.role == "admin" and await _other_active_admins(session, user.id) == 0:
        raise HTTPException(400, "This is the last active admin")

    if body.role is not None:
        user.role = body.role
    if disabling:
        user.disabled_at = datetime.now(UTC)
        # Cut the account off now rather than at the end of the access token's short lifetime where
        # we can: no new sessions (refresh) and no more prediction-API access.
        await refresh_tokens.revoke_all_for_user(session, user.id)
        await session.execute(
            sa.update(ApiKey)
            .where(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
            .values(revoked_at=sa.func.now())
        )
    elif body.disabled is False:
        user.disabled_at = None
    await session.commit()
    return AdminUserResponse(user=await _load(session, user_id))


@router.post("/{user_id}/revoke-sessions", status_code=204)
async def admin_revoke_user_sessions(user_id: uuid.UUID, session: SessionDep) -> Response:
    if await session.get(User, user_id) is None:
        raise HTTPException(404, "User not found")
    await refresh_tokens.revoke_all_for_user(session, user_id)
    await session.commit()
    return Response(status_code=204)


@router.get("/{user_id}/api-keys", response_model=ApiKeyListResponse)
async def admin_list_user_api_keys(user_id: uuid.UUID, session: SessionDep) -> ApiKeyListResponse:
    if await session.get(User, user_id) is None:
        raise HTTPException(404, "User not found")
    keys = (
        (await session.execute(sa.select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())))
        .scalars()
        .all()
    )
    return ApiKeyListResponse(keys=[ApiKeyOut.model_validate(k) for k in keys])


@router.delete("/{user_id}/api-keys/{key_id}", status_code=204)
async def admin_revoke_user_api_key(user_id: uuid.UUID, key_id: uuid.UUID, session: SessionDep) -> Response:
    """Soft revoke, like the owner's own DELETE /keys/{id}: the audit trail of the key stays."""
    res = await session.execute(
        sa.update(ApiKey)
        .where(ApiKey.id == key_id, ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .values(revoked_at=sa.func.now())
        .returning(ApiKey.id)
    )
    if res.first() is None:
        raise HTTPException(404, "API key not found")
    await session.commit()
    return Response(status_code=204)
