"""Request/response models for custom (bring-your-own) models, shared by /api/models and /api/admin/models."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import Field

from theseus.schemas.common import ApiModel

SourceKind = Literal["hub", "upload"]
Scope = Literal["global", "private"]


class CustomModelKindOut(ApiModel):
    """One sort of custom model a backend accepts, with the tasks it can be used for."""

    backend: str
    backend_label: str
    id: str
    label: str
    description: str
    modality: str
    source_kinds: list[SourceKind]
    status: Literal["stable", "experimental"]
    unavailable_reason: str | None
    tasks: list[str]


class CustomModelKindListResponse(ApiModel):
    kinds: list[CustomModelKindOut]


class CustomModelOut(ApiModel):
    id: uuid.UUID
    name: str
    description: str
    backend: str
    kind: str
    source_kind: SourceKind
    source_ref: str | None
    revision: str | None
    size_bytes: int | None
    tasks: list[str]
    # pending_upload | uploaded | validating | ready | failed
    status: str
    last_error: str | None
    enabled: bool
    archived: bool
    scope: Scope
    owner_user_id: uuid.UUID | None
    # Only filled in for admins, who see everyone's models.
    owner_email: str | None = None
    # How many training runs used it (a model in use is archived rather than deleted).
    run_count: int
    created_at: datetime
    updated_at: datetime


class CustomModelListResponse(ApiModel):
    models: list[CustomModelOut]


class CustomModelResponse(ApiModel):
    model: CustomModelOut


class CreateCustomModelBody(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    backend: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1, max_length=64)
    source_kind: SourceKind
    # The Hub repository id or timm model name. Required for a hub model, forbidden for an upload.
    source_ref: str | None = Field(default=None, max_length=200)
    tasks: list[str] = Field(min_length=1)


class UpdateCustomModelBody(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    tasks: list[str] | None = Field(default=None, min_length=1)
    enabled: bool | None = None


class UploadUrlBody(ApiModel):
    # Only the extension matters: .zip (a Hugging Face model folder) or .safetensors (a single weights file).
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class UploadUrlResponse(ApiModel):
    url: str
    key: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str]
    max_bytes: int


class DeleteCustomModelResponse(ApiModel):
    deleted: bool
    # True when runs still reference the model, so it was archived (hidden, files kept) instead.
    archived: bool
