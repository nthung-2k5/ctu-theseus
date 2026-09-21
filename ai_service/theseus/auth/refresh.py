"""Opaque, hashed, rotating refresh tokens with reuse detection.

Refresh is deliberately stateful (unlike access tokens): a fully stateless refresh token could
never be revoked, so logout would be a lie. Every transition is a guarded compare-and-swap, and
zero rows updated means someone else got there first.
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.db.models import RefreshToken
from theseus.settings import get_settings


class InvalidRefreshToken(Exception):
    pass


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def _find(session: AsyncSession, raw: str) -> RefreshToken | None:
    result = await session.execute(sa.select(RefreshToken).where(RefreshToken.token_hash == _hash(raw)))
    return result.scalar_one_or_none()


async def issue(session: AsyncSession, user_id: uuid.UUID, family_id: uuid.UUID | None = None) -> tuple[str, uuid.UUID]:
    """Create a refresh token. Returns (raw_token, row_id)."""
    raw = secrets.token_urlsafe(32)
    row = RefreshToken(
        user_id=user_id,
        family_id=family_id or uuid.uuid4(),
        token_hash=_hash(raw),
        expires_at=datetime.now(UTC) + timedelta(seconds=get_settings().refresh_token_ttl_seconds),
    )
    session.add(row)
    await session.flush()
    return raw, row.id


async def _revoke_family(session: AsyncSession, family_id: uuid.UUID) -> None:
    await session.execute(
        sa.update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=sa.func.now())
    )


async def rotate(session: AsyncSession, raw: str) -> tuple[uuid.UUID, str]:
    """Exchange a refresh token for a new one. Returns (user_id, new_raw_token).

    Presenting an already-used token means it leaked (or a client raced itself), so the whole
    family is revoked and the caller has to log in again.
    """
    row = await _find(session, raw)
    if row is None:
        raise InvalidRefreshToken("unknown token")
    if row.revoked_at is not None:
        await _revoke_family(session, row.family_id)
        await session.commit()
        raise InvalidRefreshToken("token reuse detected")
    if row.expires_at <= datetime.now(UTC):
        raise InvalidRefreshToken("expired")

    claimed = await session.execute(
        sa.update(RefreshToken)
        .where(RefreshToken.id == row.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=sa.func.now())
        .returning(RefreshToken.id)
    )
    if claimed.first() is None:
        # Lost a race with a concurrent rotation of the same token: treat it as reuse.
        await _revoke_family(session, row.family_id)
        await session.commit()
        raise InvalidRefreshToken("token reuse detected")

    new_raw, new_id = await issue(session, row.user_id, row.family_id)
    await session.execute(sa.update(RefreshToken).where(RefreshToken.id == row.id).values(replaced_by=new_id))
    return row.user_id, new_raw


async def revoke(session: AsyncSession, raw: str) -> None:
    """Logout: revoke the whole family of the presented token."""
    row = await _find(session, raw)
    if row is not None:
        await _revoke_family(session, row.family_id)
