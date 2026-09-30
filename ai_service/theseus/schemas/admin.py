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


class PluginTaskState(ApiModel):
    task: str
    task_label: str
    # Effective state for this task: a task-scoped override if one exists, else the every-task setting.
    enabled: bool
    # True when this task has its own override (as opposed to inheriting the every-task setting).
    overridden: bool


class PluginEntry(ApiModel):
    kind: str
    id: str
    label: str
    description: str = ""
    # Backend id for a built-in model; the export format group for a format.
    group: str | None = None
    available: bool = True
    unavailable_reason: str | None = None
    # The every-task switch. Tasks listed in `tasks` may override it individually.
    enabled: bool
    tasks: list[PluginTaskState]


class PluginListResponse(ApiModel):
    plugins: list[PluginEntry]


class SetPluginBody(ApiModel):
    # null clears the override, so the target inherits again.
    enabled: bool | None
    # Scope to one task; omit for every task.
    task: str | None = None


class PluginResponse(ApiModel):
    plugin: PluginEntry
