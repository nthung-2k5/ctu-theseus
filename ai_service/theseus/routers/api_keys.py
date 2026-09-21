"""/api/keys: issue, list and revoke the bearer keys the hosted prediction API (/api/v1) accepts."""

import uuid

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from theseus.auth.api_keys import generate_api_key
from theseus.db.models import ApiKey
from theseus.deps import SessionDep, UserId
from theseus.schemas.api_keys import ApiKeyListResponse, ApiKeyOut, CreateApiKeyBody, CreatedApiKey

router = APIRouter(prefix="/keys", tags=["api-keys"])


@router.post("", status_code=201, response_model=CreatedApiKey)
async def create_api_key(body: CreateApiKeyBody, user_id: UserId, session: SessionDep) -> CreatedApiKey:
    raw, key_hash, key_prefix = generate_api_key()
    key = ApiKey(user_id=user_id, name=body.name, key_hash=key_hash, key_prefix=key_prefix)
    session.add(key)
    await session.commit()
    await session.refresh(key)
    # The only moment the raw value exists: it is not derivable from the hash, so losing this
    # response means generating a new key.
    return CreatedApiKey(id=key.id, name=key.name, key_prefix=key.key_prefix, created_at=key.created_at, key=raw)


@router.get("", response_model=ApiKeyListResponse)
async def list_api_keys(user_id: UserId, session: SessionDep) -> ApiKeyListResponse:
    keys = (
        (await session.execute(sa.select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())))
        .scalars()
        .all()
    )
    return ApiKeyListResponse(keys=[ApiKeyOut.model_validate(k) for k in keys])


@router.delete("/{key_id}", status_code=204)
async def revoke_api_key(key_id: uuid.UUID, user_id: UserId, session: SessionDep) -> None:
    """Soft revoke: keeps the audit trail of what a (possibly leaked) key was."""
    res = await session.execute(
        sa.update(ApiKey)
        .where(ApiKey.id == key_id, ApiKey.user_id == user_id, ApiKey.revoked_at.is_(None))
        .values(revoked_at=sa.func.now())
        .returning(ApiKey.id)
    )
    if res.first() is None:
        raise HTTPException(404, "API key not found")
    await session.commit()
