"""Request/response models for /api/admin."""

import uuid
from datetime import datetime
from typing import Literal

from theseus.schemas.common import ApiModel


class AdminUserOut(ApiModel):
    id: uuid.UUID
    name: str
    email: str
    role: str
    disabled: bool
    created_at: datetime
    project_count: int
    run_count: int
    api_key_count: int


class AdminUserListResponse(ApiModel):
    users: list[AdminUserOut]
    total: int
    page: int
    page_size: int


class AdminUserResponse(ApiModel):
    user: AdminUserOut


class UpdateUserBody(ApiModel):
    role: Literal["user", "admin"] | None = None
    disabled: bool | None = None
