"""Register / login / refresh / logout / me.

The access token (short-lived JWT) and refresh token (opaque, hashed, rotating) both travel in
HttpOnly cookies, so a native EventSource, which cannot set an Authorization header, is
authenticated with no extra machinery.
"""

import uuid

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from theseus.auth import refresh as refresh_tokens
from theseus.auth.jwt import create_access_token
from theseus.auth.password import DUMMY_HASH, hash_password, verify_password
from theseus.db.models import User
from theseus.deps import ACCESS_COOKIE, REFRESH_COOKIE, SessionDep, UserId
from theseus.errors import envelope
from theseus.schemas.auth import LoginBody, RegisterBody, UserResponse
from theseus.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])

# Scoped to /api/auth (not just /refresh) so the browser also sends it to /logout, which must revoke it.
REFRESH_PATH = "/api/auth"


def _set_cookies(response: Response, user_id: uuid.UUID, refresh_raw: str) -> None:
    s = get_settings()
    response.set_cookie(
        ACCESS_COOKIE,
        create_access_token(user_id),
        max_age=s.access_token_ttl_seconds,
        httponly=True,
        secure=s.cookie_secure,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_raw,
        max_age=s.refresh_token_ttl_seconds,
        httponly=True,
        secure=s.cookie_secure,
        samesite="strict",
        path=REFRESH_PATH,
    )


def _clear_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH)


@router.post("/register", status_code=201, response_model=UserResponse)
async def register(body: RegisterBody, response: Response, session: SessionDep) -> UserResponse:
    user = User(
        name=body.name.strip(), email=body.email.strip().lower(), password_hash=await hash_password(body.password)
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "An account with this email already exists") from None
    await session.refresh(user)
    refresh_raw, _ = await refresh_tokens.issue(session, user.id)
    await session.commit()
    _set_cookies(response, user.id, refresh_raw)
    return UserResponse(user=user)


@router.post("/login", response_model=UserResponse)
async def login(body: LoginBody, response: Response, session: SessionDep) -> UserResponse:
    user = (await session.execute(sa.select(User).where(User.email == body.email.strip().lower()))).scalar_one_or_none()
    # Always run one verification, so response time does not reveal whether the email exists.
    ok = await verify_password(user.password_hash if user else DUMMY_HASH, body.password)
    if user is None or not ok:
        raise HTTPException(401, "Invalid email or password")
    refresh_raw, _ = await refresh_tokens.issue(session, user.id)
    await session.commit()
    _set_cookies(response, user.id, refresh_raw)
    return UserResponse(user=user)


@router.post("/refresh", status_code=204)
async def refresh_session(request: Request, session: SessionDep) -> Response:
    raw = request.cookies.get(REFRESH_COOKIE)
    if not raw:
        raise HTTPException(401, "No refresh token")
    try:
        user_id, new_raw = await refresh_tokens.rotate(session, raw)
    except refresh_tokens.InvalidRefreshToken:
        expired = JSONResponse(envelope(401, "Session expired, please sign in again"), 401)
        _clear_cookies(expired)
        return expired
    await session.commit()
    out = Response(status_code=204)
    _set_cookies(out, user_id, new_raw)
    return out


@router.post("/logout", status_code=204)
async def logout(request: Request, session: SessionDep) -> Response:
    raw = request.cookies.get(REFRESH_COOKIE)
    if raw:
        await refresh_tokens.revoke(session, raw)
        await session.commit()
    out = Response(status_code=204)
    _clear_cookies(out)
    return out


@router.get("/me", response_model=UserResponse)
async def get_me(user_id: UserId, session: SessionDep) -> UserResponse:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(401, "Account no longer exists")
    return UserResponse(user=user)
