"""Request/response models for /api/admin."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import Field

from omnitrain.schemas.common import ApiModel


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


class AdminRunOut(ApiModel):
    id: uuid.UUID
    name: str
    status: str
    project_id: uuid.UUID
    project_name: str
    task: str
    owner_email: str
    backend: str
    # The remote worker that trained (or is training) it; None for a run on this server.
    worker_name: str | None = None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failed_message: str | None


class AdminRunListResponse(ApiModel):
    runs: list[AdminRunOut]
    total: int
    page: int
    page_size: int


class AdminExportOut(ApiModel):
    id: uuid.UUID
    run_id: uuid.UUID
    run_name: str
    format: str
    status: str
    owner_email: str
    attempt: int
    max_attempts: int
    failed_message: str | None
    last_error: str | None
    created_at: datetime
    ready_at: datetime | None


class AdminExportListResponse(ApiModel):
    exports: list[AdminExportOut]
    total: int
    page: int
    page_size: int


class LaneOut(ApiModel):
    # train | export | validate
    name: str
    label: str
    # Waiting to be claimed.
    queued: int
    # In flight right now / how many may run at once. None when no dispatcher is running (never in production).
    running: int | None
    capacity: int | None


class BackendStatusOut(ApiModel):
    id: str
    label: str
    available: bool
    unavailable_reason: str | None


class SystemCounts(ApiModel):
    users: int
    admins: int
    disabled_users: int
    projects: int
    runs: int
    active_runs: int
    exports: int
    custom_models: int
    custom_model_bytes: int
    # Remote training workers that are not revoked, and how many of those were heard from recently.
    workers: int
    workers_online: int


class SystemResponse(ApiModel):
    counts: SystemCounts
    lanes: list[LaneOut]
    backends: list[BackendStatusOut]


class CreateWorkerBody(ApiModel):
    name: str = Field(min_length=1, max_length=100)


class UpdateWorkerBody(ApiModel):
    enabled: bool


class WorkerOut(ApiModel):
    id: uuid.UUID
    name: str
    enabled: bool
    # Heard from within the server's `worker_online_seconds`. A revoked worker is never online.
    online: bool
    last_seen_at: datetime | None
    created_at: datetime
    revoked_at: datetime | None
    # What the worker last said about itself (empty until its first call).
    hostname: str | None
    gpu: str | None
    version: str | None
    backends: list[str]
    current_run_id: uuid.UUID | None
    current_run_name: str | None


class WorkerListResponse(ApiModel):
    workers: list[WorkerOut]


class CreatedWorker(ApiModel):
    worker: WorkerOut
    # The raw token: returned exactly once, never stored, not derivable from its hash.
    token: str
