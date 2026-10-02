"""/api/models: a user's own bring-your-own models.

Each route only ever sees models the caller owns (`custom_model_ops.load` answers 404 for anyone else's).
Admins manage global models, and see everyone's, under /api/admin/models.
"""

import uuid

from fastapi import APIRouter

from theseus.deps import SessionDep, UserId
from theseus.schemas.custom_models import (
    CreateCustomModelBody,
    CustomModelKindListResponse,
    CustomModelListResponse,
    CustomModelResponse,
    DeleteCustomModelResponse,
    UpdateCustomModelBody,
    UploadUrlBody,
    UploadUrlResponse,
)
from theseus.services import custom_model_ops as ops

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/kinds", response_model=CustomModelKindListResponse)
async def list_custom_model_kinds(_: UserId) -> CustomModelKindListResponse:
    """What can be added, per backend and task: the options of the "add a model" form."""
    return CustomModelKindListResponse(kinds=ops.list_kinds())


@router.get("", response_model=CustomModelListResponse)
async def list_my_models(user_id: UserId, session: SessionDep) -> CustomModelListResponse:
    return CustomModelListResponse(models=await ops.list_models(session, owner_user_id=user_id))


@router.post("", status_code=201, response_model=CustomModelResponse)
async def create_my_model(body: CreateCustomModelBody, user_id: UserId, session: SessionDep) -> CustomModelResponse:
    row = await ops.create(session, body, user_id, global_=False)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=False))


@router.patch("/{model_id}", response_model=CustomModelResponse)
async def update_my_model(
    model_id: uuid.UUID, body: UpdateCustomModelBody, user_id: UserId, session: SessionDep
) -> CustomModelResponse:
    row = await ops.load(session, model_id, user_id, is_admin=False)
    row = await ops.update(session, row, body)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=False))


@router.delete("/{model_id}", response_model=DeleteCustomModelResponse)
async def delete_my_model(model_id: uuid.UUID, user_id: UserId, session: SessionDep) -> DeleteCustomModelResponse:
    row = await ops.load(session, model_id, user_id, is_admin=False)
    deleted, archived = await ops.delete(session, row)
    return DeleteCustomModelResponse(deleted=deleted, archived=archived)


@router.post("/{model_id}/upload-url", response_model=UploadUrlResponse)
async def request_model_upload_url(
    model_id: uuid.UUID, body: UploadUrlBody, user_id: UserId, session: SessionDep
) -> UploadUrlResponse:
    """A short-lived presigned PUT URL: the browser uploads the weights straight to object storage."""
    row = await ops.load(session, model_id, user_id, is_admin=False)
    return UploadUrlResponse(**await ops.upload_url(session, row, body))


@router.post("/{model_id}/finalize", response_model=CustomModelResponse)
async def finalize_model_upload(model_id: uuid.UUID, user_id: UserId, session: SessionDep) -> CustomModelResponse:
    row = await ops.load(session, model_id, user_id, is_admin=False)
    row = await ops.finalize_upload(session, row)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=False))


@router.post("/{model_id}/retry", response_model=CustomModelResponse)
async def retry_model_validation(model_id: uuid.UUID, user_id: UserId, session: SessionDep) -> CustomModelResponse:
    row = await ops.load(session, model_id, user_id, is_admin=False)
    row = await ops.retry(session, row)
    return CustomModelResponse(model=await ops.describe(session, row, with_owner=False))
