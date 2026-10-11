"""A user's saved credentials for outside services: today, one Hugging Face access token.

The token lets the user import models their Hugging Face account can read but the public cannot (private
repositories, and gated ones they have been granted). It is checked against Hugging Face when it is saved,
stored encrypted (`services/secrets.py`), and only ever used for models the SAME user imported: a global
model, which an admin makes available to everyone, never carries anyone's token.
"""

import asyncio
import logging
import uuid

import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import UserIntegration
from omnitrain.schemas.integrations import HuggingFaceStatus
from omnitrain.services import secrets

logger = logging.getLogger(__name__)

HUGGINGFACE = "huggingface"
_HINT_CHARS = 4


def _status(row: UserIntegration | None) -> HuggingFaceStatus:
    if row is None:
        return HuggingFaceStatus(connected=False)
    return HuggingFaceStatus(
        connected=True, account_name=row.account_name, token_hint=row.token_hint, connected_at=row.updated_at
    )


async def _row(session: AsyncSession, user_id: uuid.UUID) -> UserIntegration | None:
    return await session.get(UserIntegration, (user_id, HUGGINGFACE))


async def huggingface_status(session: AsyncSession, user_id: uuid.UUID) -> HuggingFaceStatus:
    return _status(await _row(session, user_id))


def _check_token(token: str) -> str | None:
    """Ask Hugging Face who the token belongs to. Raises HTTPException for a token it refuses or cannot be asked."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = HfApi().whoami(token=token)
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise HTTPException(
                400, "Hugging Face did not accept this token. Check that it is correct and not revoked"
            ) from None
        raise HTTPException(
            502, "Hugging Face could not be reached to check the token. Try again in a moment"
        ) from None
    except Exception:  # noqa: BLE001  a network failure: the token itself may be fine
        logger.warning("Could not reach Hugging Face to check a token", exc_info=True)
        raise HTTPException(
            502, "Hugging Face could not be reached to check the token. Try again in a moment"
        ) from None
    name = info.get("name") if isinstance(info, dict) else None
    return name if isinstance(name, str) else None


async def connect_huggingface(session: AsyncSession, user_id: uuid.UUID, token: str) -> HuggingFaceStatus:
    """Verify and save (or replace) the user's token."""
    account_name = await asyncio.to_thread(_check_token, token)
    row = await _row(session, user_id)
    if row is None:
        row = UserIntegration(user_id=user_id, provider=HUGGINGFACE)
        session.add(row)
    row.token_encrypted = secrets.encrypt(token)
    row.token_hint = token[-_HINT_CHARS:]
    row.account_name = account_name
    row.updated_at = sa.func.now()
    await session.commit()
    await session.refresh(row)
    return _status(row)


async def disconnect_huggingface(session: AsyncSession, user_id: uuid.UUID) -> None:
    await session.execute(
        sa.delete(UserIntegration).where(UserIntegration.user_id == user_id, UserIntegration.provider == HUGGINGFACE)
    )
    await session.commit()


async def huggingface_token(session: AsyncSession, user_id: uuid.UUID | None) -> str | None:
    """The user's saved token, or None when they have none (or it can no longer be decrypted)."""
    if user_id is None:
        return None
    row = await _row(session, user_id)
    return secrets.decrypt(row.token_encrypted) if row is not None else None


async def huggingface_token_for(user_id: uuid.UUID | None) -> str | None:
    """Same, for the background jobs that have no request session."""
    if user_id is None:
        return None
    async with get_sessionmaker()() as session:
        return await huggingface_token(session, user_id)
