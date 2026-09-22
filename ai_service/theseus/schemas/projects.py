"""Request/response models for projects, dataset versions and label classes.

Shapes mirror what the web app consumed from the Elysia gateway (web/store/types.ts), so the
frontend migration is a type swap rather than a behavior change.
"""

import uuid
from datetime import datetime

from pydantic import Field

from theseus.augmentation.config import AugmentationConfig
from theseus.db.enums import DatasetModality, DatasetVersionStatus, ProjectTask, SplitType
from theseus.schemas.common import ApiModel

HEX_COLOR = r"^#[0-9a-fA-F]{6}$"


class LabelClassOut(ApiModel):
    class_id: uuid.UUID
    dataset_id: uuid.UUID
    name: str
    description: str | None
    ui_color_hex: str | None
    is_active: bool
    created_at: datetime | None


class SplitCounts(ApiModel):
    train: int = 0
    validation: int = 0
    test: int = 0


class DatasetSplit(ApiModel):
    split_type: SplitType
    item_count: int


class VersionOut(ApiModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_tag: str | None
    status: DatasetVersionStatus
    # Live membership total, derived from the split counts (the column is only written at snapshot time).
    item_count: int | None
    class_count: int | None
    failed_message: str | None
    parquet_key: str | None
    # What the snapshot was built with (None: no augmentation), and how many augmented copies it holds.
    augmentation_config: AugmentationConfig | None
    augmented_count: int
    built_at: datetime | None
    created_at: datetime
    split_counts: SplitCounts | None = None
    # GET /versions/{id} returns computed `splits` instead.
    splits: list[DatasetSplit] | None = None


class DatasetOut(ApiModel):
    project_id: uuid.UUID
    modality: DatasetModality
    created_at: datetime
    updated_at: datetime
    draft: VersionOut | None
    versions: list[VersionOut]
    classes: list[LabelClassOut]


class DraftDatasetOut(ApiModel):
    modality: DatasetModality


class ProjectSummary(ApiModel):
    id: uuid.UUID
    name: str
    description: str | None
    task: ProjectTask
    created_at: datetime
    draft_dataset: DraftDatasetOut | None = None


class ProjectRow(ApiModel):
    id: uuid.UUID
    name: str
    description: str | None
    task: ProjectTask
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectRow):
    run_count: int
    version_count: int
    dataset: DatasetOut | None


class ProjectListResponse(ApiModel):
    projects: list[ProjectSummary]


class ProjectResponse(ApiModel):
    project: ProjectRow


class ProjectDetailResponse(ApiModel):
    project: ProjectDetail


class CreateProjectBody(ApiModel):
    name: str = Field(min_length=1)
    description: str | None = None
    task: ProjectTask


class UpdateProjectBody(ApiModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None


class ClassListResponse(ApiModel):
    classes: list[LabelClassOut]


class ClassResponse(ApiModel):
    # `class` is a Python keyword, so the attribute is class_ and only the wire name is "class".
    class_: LabelClassOut = Field(alias="class")


class CreateClassBody(ApiModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    ui_color_hex: str | None = Field(default=None, pattern=HEX_COLOR)


class UpdateClassBody(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    ui_color_hex: str | None = Field(default=None, pattern=HEX_COLOR)
