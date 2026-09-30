"""/api/admin/models: the global custom-model catalog, and every user's private models.

Same operations as /api/models (see services/custom_model_ops.py), without the owner restriction. A model
created here has no owner, so every user can train on it wherever its `tasks` say it is offered.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from theseus.deps import AdminId, SessionDep
from theseus.schemas.custom_models import (
    CreateCustomModelBody,
    CustomModelListResponse,
    CustomModelResponse,
    DeleteCustomModelResponse,
    UpdateCustomModelBody,
    UploadUrlBody,
    UploadUrlResponse,
)
from theseus.services import custom_model_ops as ops

router = APIRouter(prefix="/models")


@router.get("", response_model=CustomModelListResponse)
async def admin_list_models(
    session: SessionDep,
    scope: Annotated[str | None, Query(pattern="^(global|private)$")] = None,
    status: Annotated[str | None, Query(pattern="^(pending_upload|uploaded|validating|ready|failed)$")] = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
    include_archived: bool = False,
) -> CustomModelListResponse:
    return CustomModelListResponse(
        models=await ops.list_models(
            session, scope=scope, status=status, q=q, include_archived=include_archived, with_owner=True
        )
    )


@router.post("", status_code=201, response_model=CustomModelResponse)
async def admin_create_model(
    body: CreateCustomModelBody, admin_id: AdminId, session: SessionDep
) -> CustomModelResponse:
    """Create a GLOBAL model, available to every user."""
    row = await ops.create(session, body, admin_id, global_=True)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=True))


@router.patch("/{model_id}", response_model=CustomModelResponse)
async def admin_update_model(
    model_id: uuid.UUID, body: UpdateCustomModelBody, admin_id: AdminId, session: SessionDep
) -> CustomModelResponse:
    row = await ops.load(session, model_id, admin_id, is_admin=True)
    row = await ops.update(session, row, body)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=True))


@router.delete("/{model_id}", response_model=DeleteCustomModelResponse)
async def admin_delete_model(model_id: uuid.UUID, admin_id: AdminId, session: SessionDep) -> DeleteCustomModelResponse:
    row = await ops.load(session, model_id, admin_id, is_admin=True)
    deleted, archived = await ops.delete(session, row)
    return DeleteCustomModelResponse(deleted=deleted, archived=archived)


@router.post("/{model_id}/upload-url", response_model=UploadUrlResponse)
async def admin_model_upload_url(
    model_id: uuid.UUID, body: UploadUrlBody, admin_id: AdminId, session: SessionDep
) -> UploadUrlResponse:
    row = await ops.load(session, model_id, admin_id, is_admin=True)
    return UploadUrlResponse(**await ops.upload_url(session, row, body))


@router.post("/{model_id}/finalize", response_model=CustomModelResponse)
async def admin_finalize_model_upload(
    model_id: uuid.UUID, admin_id: AdminId, session: SessionDep
) -> CustomModelResponse:
    row = await ops.load(session, model_id, admin_id, is_admin=True)
    row = await ops.finalize_upload(session, row)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=True))


@router.post("/{model_id}/retry", response_model=CustomModelResponse)
async def admin_retry_model_validation(
    model_id: uuid.UUID, admin_id: AdminId, session: SessionDep
) -> CustomModelResponse:
    row = await ops.load(session, model_id, admin_id, is_admin=True)
    row = await ops.retry(session, row)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=True))
