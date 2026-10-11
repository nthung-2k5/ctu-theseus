"""Request/response models for custom (bring-your-own) models, shared by /api/models and /api/admin/models."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from omnitrain.schemas.common import ApiModel

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
    # False for a kind whose framework fetches its own weights by name (a timm model): nothing to inspect.
    needs_files: bool
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
    # Every task this model is offered for: all the tasks its kind serves.
    tasks: list[str]
    # pending_upload | uploaded | validating | ready | failed
    status: str
    last_error: str | None
    enabled: bool
    archived: bool
    # Fetched with its owner's Hugging Face token (a private or gated repository).
    private: bool = False
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


class InspectModelBody(ApiModel):
    """What to look at: a Hub repository by id, or a `config.json` the browser read out of an uploaded zip.
    Exactly one of the two."""

    source_ref: str | None = Field(default=None, min_length=1, max_length=200)
    config: dict[str, Any] | None = None
    # The model is being made for everyone (an admin's form). A shared model never carries one person's
    # Hugging Face access, so it is looked up anonymously and only a public repository passes.
    shared: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> "InspectModelBody":
        if (self.source_ref is None) == (self.config is None):
            raise ValueError("Send either sourceRef or config")
        return self


class InspectModelResponse(ApiModel):
    """What checking a model before importing it found out (creates nothing). `source_ref`, `revision` and
    `size_bytes` are only known for a Hub repository; a config read from an upload has none of them."""

    source_ref: str | None
    revision: str | None
    size_bytes: int | None
    model_type: str | None
    architectures: list[str]
    # The kind config.json points to; null when it does not say.
    suggested_backend: str | None
    suggested_kind: str | None
    # Errors mean the model cannot be imported; warnings do not.
    errors: list[str]
    warnings: list[str]
    # Private or gated on Hugging Face: only readable with the importer's own saved token.
    private: bool = False


class UpdateCustomModelBody(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    enabled: bool | None = None


class UploadUrlBody(ApiModel):
    # Only the extension matters: it must be a .zip of a model folder (config.json plus the weights).
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
