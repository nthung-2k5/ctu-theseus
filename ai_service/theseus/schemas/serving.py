"""Request/response models for inference and export."""

import uuid
from datetime import datetime
from typing import Any

from theseus.db.enums import ExportFormat, ExportLang, ExportStatus, ExportTier
from theseus.schemas.common import ApiModel


class InferenceAccepted(ApiModel):
    inference_id: uuid.UUID


class PolledJob(ApiModel):
    """pending | success (with output) | batch (with rowCount) | failed (with error).

    Fields that do not apply to a state are omitted from the response (exclude_none), which is
    the same discriminated shape the gateway returned.
    """

    status: str
    output: Any | None = None
    row_count: int | None = None
    error: str | None = None


class SyncPredictResponse(PolledJob):
    inference_id: uuid.UUID


class InferenceJobRow(ApiModel):
    id: uuid.UUID
    run_id: uuid.UUID
    status: str
    output: Any | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class InferenceJobListResponse(ApiModel):
    jobs: list[InferenceJobRow]


class CreateExportBody(ApiModel):
    tier: ExportTier
    format: ExportFormat
    lang: ExportLang | None = None


class ExportAccepted(ApiModel):
    export_id: uuid.UUID


class ExportRow(ApiModel):
    id: uuid.UUID
    run_id: uuid.UUID
    user_id: uuid.UUID
    tier: ExportTier
    format: ExportFormat
    lang: ExportLang | None
    status: ExportStatus
    bundle_key: str | None
    byte_size: int | None
    checksum: str | None
    failed_message: str | None
    created_at: datetime
    ready_at: datetime | None
    updated_at: datetime


class ExportListResponse(ApiModel):
    exports: list[ExportRow]


class ExportResponse(ApiModel):
    # `export` is not a reserved word in Python, but keep the wire name explicit and stable.
    export: ExportRow
