"""Request/response models for training runs, evaluation and sweeps."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from theseus.db.enums import (
    DatasetModality,
    DatasetVersionStatus,
    EvaluationSplit,
    EvaluationStatus,
    SplitType,
    SweepStatus,
    SweepStrategy,
    TrainingStatus,
)
from theseus.schemas.common import ApiModel
from theseus.services.ludwig_config import TrainerSelections


class EvaluationBrief(ApiModel):
    """Denormalized accuracy / macro-F1, so lists and comparisons need no per-run round trip."""

    status: EvaluationStatus
    accuracy: float | None
    macro_f1: float | None


class RunSummary(ApiModel):
    id: uuid.UUID
    name: str
    status: TrainingStatus
    failed_message: str | None
    completed_at: datetime | None
    started_at: datetime | None
    created_at: datetime
    dataset_version_id: uuid.UUID
    evaluation: EvaluationBrief | None = None


class RunListResponse(ApiModel):
    runs: list[RunSummary]


class RunRow(ApiModel):
    id: uuid.UUID
    name: str
    status: TrainingStatus
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    sweep_id: uuid.UUID | None
    trial_index: int | None
    hyperparameters: Any
    best_epoch: int | None
    started_at: datetime | None
    completed_at: datetime | None
    failed_message: str | None
    created_at: datetime
    updated_at: datetime


class RunCreatedResponse(ApiModel):
    run: RunRow


class MetricRow(ApiModel):
    training_run_id: uuid.UUID
    epoch: int
    split: SplitType
    metric_name: str
    metric_value: float
    created_at: datetime


class RunVersionDataset(ApiModel):
    project_id: uuid.UUID
    modality: DatasetModality


class RunVersion(ApiModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_tag: str | None
    status: DatasetVersionStatus
    item_count: int | None
    class_count: int | None
    failed_message: str | None
    parquet_key: str | None
    built_at: datetime | None
    created_at: datetime
    dataset: RunVersionDataset


class RunDetail(ApiModel):
    id: uuid.UUID
    name: str
    status: TrainingStatus
    hyperparameters: Any
    failed_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    dataset_version: RunVersion
    metrics: list[MetricRow]


class RunDetailResponse(ApiModel):
    run: RunDetail


class TrainBody(ApiModel):
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: uuid.UUID
    hyperparameters: TrainerSelections | None = None


class RunStatusResponse(ApiModel):
    status: TrainingStatus
    epochs_total: int | None
    latest_metrics: list[MetricRow]


class EvaluationOut(ApiModel):
    run_id: uuid.UUID
    status: EvaluationStatus
    split: EvaluationSplit | None
    report_key: str | None
    predictions_key: str | None
    report: Any | None
    accuracy: float | None
    macro_f1: float | None
    failed_message: str | None
    evaluated_at: datetime


class EvaluationResponse(ApiModel):
    evaluation: EvaluationOut


class ErrorItem(ApiModel):
    id: uuid.UUID
    text: str | None
    download_url: str | None


class EvaluationErrorRow(ApiModel):
    item_id: str
    actual: str
    predicted: str
    confidence: float | None = None
    item: ErrorItem | None


class EvaluationErrorsResponse(ApiModel):
    errors: list[EvaluationErrorRow]
    total: int
    page: int
    per_page: int


# -- Sweeps ----------------------------------------------------------------------------------


class SearchSpace(ApiModel):
    """One candidate-value list per sweepable trainer knob (camelCase keys, as stored and expanded)."""

    epochs: list[int] | None = Field(default=None, min_length=1)
    batch_size: list[int | Literal["auto"]] | None = Field(default=None, min_length=1)
    learning_rate: list[float] | None = Field(default=None, min_length=1)
    early_stop_patience: list[int] | None = Field(default=None, min_length=1)
    encoder_id: list[str] | None = Field(default=None, min_length=1)


class CreateSweepBody(ApiModel):
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: uuid.UUID
    search_space: SearchSpace
    strategy: SweepStrategy
    max_trials: int = Field(ge=1, le=50)


class SweepRow(ApiModel):
    id: uuid.UUID
    project_id: uuid.UUID
    dataset_version_id: uuid.UUID
    name: str
    search_space: Any
    strategy: SweepStrategy
    max_trials: int
    status: SweepStatus
    created_at: datetime
    updated_at: datetime


class SweepCreatedResponse(ApiModel):
    sweep: SweepRow
    trials: list[RunRow]


class SweepSummary(ApiModel):
    id: uuid.UUID
    name: str
    strategy: SweepStrategy
    max_trials: int
    status: SweepStatus
    created_at: datetime
    trial_count: int
    completed_trial_count: int


class SweepListResponse(ApiModel):
    sweeps: list[SweepSummary]


class SweepTrial(ApiModel):
    id: uuid.UUID
    name: str
    status: TrainingStatus
    hyperparameters: Any
    trial_index: int | None
    failed_message: str | None
    created_at: datetime
    completed_at: datetime | None
    evaluation: EvaluationBrief | None = None


class SweepDetailResponse(ApiModel):
    sweep: SweepRow
    trials: list[SweepTrial]
