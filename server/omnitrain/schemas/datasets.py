"""Request/response models for the dataset pool: items, annotations, splits, health."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field, model_validator

from omnitrain.db.enums import (
    AnnotationType,
    AudioCodec,
    DatasetModality,
    ImageFormat,
    SplitType,
    VideoContainer,
)
from omnitrain.domain.boxes import validate_box_annotation
from omnitrain.domain.token_tags import tag_shape_error
from omnitrain.plugins.augmentation.config import AugmentationConfig, AugmentationInfo
from omnitrain.plugins.preprocessing.config import PreprocessingConfig, PreprocessingInfo
from omnitrain.schemas.common import ApiModel
from omnitrain.schemas.projects import VersionOut


class VersionDataset(ApiModel):
    project_id: uuid.UUID
    modality: DatasetModality


class VersionDetail(VersionOut):
    dataset: VersionDataset


class CreateVersionBody(ApiModel):
    version_tag: str = Field(min_length=1, max_length=50)
    # Replace items with a deterministic preprocessed copy, per split. Omit for no preprocessing.
    preprocessing: PreprocessingConfig | None = None
    # Add augmented copies of the train split to the snapshot. Omit for a plain snapshot.
    augmentation: AugmentationConfig | None = None


class PreprocessingListResponse(ApiModel):
    preprocessing: list[PreprocessingInfo]


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


class VideoFeaturesOut(ApiModel):
    item_id: uuid.UUID
    width: int
    height: int
    duration_seconds: Decimal
    fps: Decimal
    frame_count: int
    container: VideoContainer | None
    codec: str | None


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
    video_features: VideoFeaturesOut | None = None
    tabular_features: TabularFeaturesOut | None = None
    annotations: list[AnnotationOut] = []
    split_type: SplitType
    download_url: str | None = None
    # A small JPEG of an image item (a few KB, against a full-size download), for grids. Authenticated like
    # the rest of the API; it redirects to object storage. None for anything that isn't an image.
    thumbnail_url: str | None = None
    # Set on any derived item (preprocessed or augmented): the pool original it came from.
    source_item_id: uuid.UUID | None = None
    source_external_id: str | None = None
    # The preprocessing ops (if any) that produced this item, carried forward onto an augmented copy.
    preprocessing: Any | None = None
    # The augmentation ops that produced this item; set only on an augmented copy.
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

    @model_validator(mode="after")
    def _a_box_is_well_formed(self) -> "AnnotationIn":
        # One annotation is one box (or the "no objects" marker); see domain/boxes.py for the convention.
        if self.annotation_type == "bounding_box":
            validate_box_annotation(self.class_id, self.label_structured)
        # Token tags are one tag per word; whether the count matches the item's text is checked where the text is known.
        if self.annotation_type == "token_tags" and (problem := tag_shape_error(self.label_structured)):
            raise ValueError(problem)
        return self


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


class PreflightBody(ApiModel):
    """SHA-256 digests (hex) of files the client is about to upload."""

    hashes: list[str] = Field(max_length=5000)


class PreflightResponse(ApiModel):
    """The digests whose content is already a live item of the draft: sending them again changes nothing."""

    existing: list[str]


class ItemFilter(ApiModel):
    """The item listing's filters, to act on "everything matching" instead of on listed ids."""

    split: SplitType | None = None
    # A class id, or 'unassigned'.
    class_id: str | None = None
    search: str | None = None
    labeled: bool | None = None


class ItemSelection(ApiModel):
    """Which draft items a bulk operation acts on: explicit ids, or a filter minus some exceptions."""

    item_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=1000)
    filter: ItemFilter | None = None
    # Items to leave out of a filter match, e.g. rows the user un-ticked after "select all matching".
    exclude_item_ids: list[uuid.UUID] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> "ItemSelection":
        if (self.item_ids is None) == (self.filter is None):
            raise ValueError("Send exactly one of itemIds or filter")
        if self.item_ids is not None and self.exclude_item_ids:
            raise ValueError("excludeItemIds only applies together with a filter")
        return self


class DeleteItemsBody(ItemSelection):
    pass


class DeleteOutcome(ApiModel):
    item_id: uuid.UUID
    outcome: str


class DeleteItemsResponse(ApiModel):
    results: list[DeleteOutcome]


class SetSplitBody(ItemSelection):
    split: SplitType


class SetSplitResponse(ApiModel):
    updated: list[uuid.UUID]


class ClassifyBody(ItemSelection):
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


class VideoHealth(ApiModel):
    count: int
    duration_seconds: MinMaxAvg
    width: MinMaxAvg
    height: MinMaxAvg
    fps: MinMaxAvg
    containers: dict[str, int]


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
    video: VideoHealth | None
    text: TextHealth | None
    tabular: TabularHealth | None


class DatasetHealthResponse(ApiModel):
    health: DatasetHealth
