"""Request/response models for the dataset pool: items, annotations, splits, health."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from theseus.augmentation.config import AugmentationConfig, AugmentationInfo
from theseus.db.enums import (
    AnnotationType,
    AudioCodec,
    DatasetModality,
    ImageFormat,
    SplitType,
)
from theseus.schemas.common import ApiModel
from theseus.schemas.projects import VersionOut


class VersionDataset(ApiModel):
    project_id: uuid.UUID
    modality: DatasetModality


class VersionDetail(VersionOut):
    dataset: VersionDataset


class CreateVersionBody(ApiModel):
    version_tag: str = Field(min_length=1, max_length=50)
    # Add augmented copies of the train split to the snapshot. Omit for a plain snapshot.
    augmentation: AugmentationConfig | None = None


class AugmentationListResponse(ApiModel):
    augmentations: list[AugmentationInfo]


class VersionCreatedResponse(ApiModel):
    version: VersionOut


class VersionDetailResponse(ApiModel):
    version: VersionDetail


# -- Item output -----------------------------------------------------------------------------


class AnnotationOut(ApiModel):
    id: uuid.UUID
    item_id: uuid.UUID
    annotator_id: str | None
    annotation_type: AnnotationType
    class_id: uuid.UUID | None
    label_text_sequence: str | None
    label_structured: Any | None
    confidence_score: Decimal | None
    created_at: datetime | None


class TextFeaturesOut(ApiModel):
    item_id: uuid.UUID
    raw_text: str
    token_count: int | None
    language_code: str | None
    meta_json: Any | None


class VisionFeaturesOut(ApiModel):
    item_id: uuid.UUID
    width: int
    height: int
    channels: int | None
    image_format: ImageFormat | None
    exif_data: Any | None


class AudioFeaturesOut(ApiModel):
    item_id: uuid.UUID
    duration_seconds: Decimal
    sample_rate_hz: int
    channels: int | None
    audio_codec: AudioCodec | None


class TabularFeaturesOut(ApiModel):
    item_id: uuid.UUID
    features_json: Any


class ItemRow(ApiModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    external_id: str | None
    storage_url: str | None
    content_hash: str | None
    byte_size: int | None
    created_at: datetime
    deleted_at: datetime | None


class ItemOut(ItemRow):
    text_features: TextFeaturesOut | None = None
    vision_features: VisionFeaturesOut | None = None
    audio_features: AudioFeaturesOut | None = None
    tabular_features: TabularFeaturesOut | None = None
    annotations: list[AnnotationOut] = []
    split_type: SplitType
    download_url: str | None = None
    # Set on augmented copies: the original they came from and the ops that produced them.
    source_item_id: uuid.UUID | None = None
    source_external_id: str | None = None
    augmentation: Any | None = None


class ClassCount(ApiModel):
    class_id: uuid.UUID
    count: int


class ItemListResponse(ApiModel):
    items: list[ItemOut]
    total: int
    labeled_count: int
    class_counts: list[ClassCount]
    unassigned_count: int
    page: int
    per_page: int


# -- Item input ------------------------------------------------------------------------------


class TextFeaturesIn(ApiModel):
    raw_text: str
    token_count: float | None = None
    language_code: str | None = None
    meta_json: Any | None = None


class VisionFeaturesIn(ApiModel):
    width: float
    height: float
    channels: float | None = None
    image_format: ImageFormat | None = None
    exif_data: Any | None = None


class AudioFeaturesIn(ApiModel):
    duration_seconds: float
    sample_rate_hz: float
    channels: float | None = None
    audio_codec: AudioCodec | None = None


class TabularFeaturesIn(ApiModel):
    features_json: Any


class AnnotationIn(ApiModel):
    annotator_id: str | None = None
    annotation_type: AnnotationType
    class_id: uuid.UUID | None = None
    label_text_sequence: str | None = None
    label_structured: Any | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=1)


class ItemIn(ApiModel):
    """No storage_url here on purpose: this route only creates items from inline data (text/tabular).

    Accepting a client-chosen S3 key would let one tenant point an item at another tenant pool
    object. The only path that writes storage_url is the upload route, which derives the key
    server-side from the content hash.
    """

    split: SplitType
    external_id: str | None = None
    text_features: TextFeaturesIn | None = None
    vision_features: VisionFeaturesIn | None = None
    audio_features: AudioFeaturesIn | None = None
    tabular_features: TabularFeaturesIn | None = None
    annotations: list[AnnotationIn] | None = None


class CreateItemsBody(ApiModel):
    items: list[ItemIn] = Field(min_length=1)


class ItemFailure(ApiModel):
    index: int
    message: str


class CreateItemsResponse(ApiModel):
    created: list[ItemRow]
    failed: list[ItemFailure]


class ItemRowWithDuplicate(ItemRow):
    is_duplicate: bool


class UploadResult(ApiModel):
    """One file outcome, in the same {status, value | reason} shape the gateway returned."""

    status: str
    value: ItemRowWithDuplicate | None = None
    reason: str | None = None


class UploadResponse(ApiModel):
    results: list[UploadResult]


class DeleteItemsBody(ApiModel):
    item_ids: list[uuid.UUID] = Field(min_length=1, max_length=1000)


class DeleteOutcome(ApiModel):
    item_id: uuid.UUID
    outcome: str


class DeleteItemsResponse(ApiModel):
    results: list[DeleteOutcome]


class SetSplitBody(ApiModel):
    item_ids: list[uuid.UUID] = Field(min_length=1, max_length=1000)
    split: SplitType


class SetSplitResponse(ApiModel):
    updated: list[uuid.UUID]


class ClassifyBody(ApiModel):
    item_ids: list[uuid.UUID] = Field(min_length=1, max_length=1000)
    class_id: uuid.UUID


class ClassifyResponse(ApiModel):
    updated: int
    failed: int


class Ratios(ApiModel):
    train: float = Field(ge=0)
    validation: float = Field(ge=0)
    test: float = Field(ge=0)


class AutoSplitBody(ApiModel):
    stratify: bool | None = None
    ratios: Ratios | None = None


class AutoSplitCounts(ApiModel):
    train: int
    validation: int
    test: int


class AutoSplitResponse(ApiModel):
    updated: int
    stratified: bool | None = None
    splits: AutoSplitCounts | None = None
    warnings: list[str] | None = None


# -- Annotations -----------------------------------------------------------------------------


class CreateAnnotationBody(AnnotationIn):
    pass


class UpdateAnnotationBody(ApiModel):
    class_id: uuid.UUID | None = None
    label_text_sequence: str | None = None
    label_structured: Any | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=1)


class AnnotationResponse(ApiModel):
    annotation: AnnotationOut


class AnnotationListResponse(ApiModel):
    annotations: list[AnnotationOut]


# -- Health ----------------------------------------------------------------------------------


class HealthClassCount(ApiModel):
    class_id: uuid.UUID
    name: str
    count: int


class MinMaxAvg(ApiModel):
    min: float
    max: float
    avg: float


class VisionHealth(ApiModel):
    count: int
    width: MinMaxAvg
    height: MinMaxAvg
    formats: dict[str, int]


class AudioHealth(ApiModel):
    count: int
    duration_seconds: MinMaxAvg
    sample_rates: dict[str, int]


class TextHealth(ApiModel):
    count: int
    token_count: MinMaxAvg | None
    languages: dict[str, int]


class TabularHealth(ApiModel):
    count: int


class DatasetHealth(ApiModel):
    item_count: int
    labeled_count: int
    unlabeled_count: int
    modality: DatasetModality
    class_distribution: list[HealthClassCount]
    small_classes: list[HealthClassCount]
    duplicate_content_hashes: int
    missing_content_hash: int
    vision: VisionHealth | None
    audio: AudioHealth | None
    text: TextHealth | None
    tabular: TabularHealth | None


class DatasetHealthResponse(ApiModel):
    health: DatasetHealth
