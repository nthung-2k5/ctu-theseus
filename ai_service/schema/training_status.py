from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class TrainingMetric(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    loss: Annotated[float, Field(ge=0.0)]
    accuracy: Annotated[float, Field(ge=0.0, le=1.0)]


class TrainingProgress(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: UUID
    epoch: Annotated[int, Field(ge=0)]
    train_metrics: TrainingMetric
    validation_metrics: TrainingMetric
    test_metrics: TrainingMetric


class BaseTrainingStatus(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: UUID
    type: Literal["train"]
    timestamp: datetime | None


class TrainingStartedStatus(BaseTrainingStatus):
    status: Literal["training"]


class TrainingCompletedStatus(BaseTrainingStatus):
    status: Literal["completed"]


class TrainingFailedStatus(BaseTrainingStatus):
    status: Literal["failed"]
    error_message: str


TrainingStatus = TrainingStartedStatus | TrainingCompletedStatus | TrainingFailedStatus
