import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db import enums as e
from theseus.db.base import Base
from theseus.db.models._common import created_at, pg_enum, updated_at, uuid_pk

project_task_t = pg_enum(e.ProjectTask, "project_task")
modality_t = pg_enum(e.DatasetModality, "modality")
split_type_t = pg_enum(e.SplitType, "split_type")
image_format_t = pg_enum(e.ImageFormat, "image_format")
audio_codec_t = pg_enum(e.AudioCodec, "audio_codec")
annotation_type_t = pg_enum(e.AnnotationType, "annotation_type")
dataset_version_status_t = pg_enum(e.DatasetVersionStatus, "dataset_version_status")

CASCADE = "CASCADE"


def _fk(target: str, ondelete: str = CASCADE, **kw: Any) -> Mapped[uuid.UUID]:
    return mapped_column(sa.ForeignKey(target, ondelete=ondelete), **kw)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = _fk("users.id", index=True)
    name: Mapped[str] = mapped_column(sa.Text)
    description: Mapped[str | None] = mapped_column(sa.Text, server_default="")
    task: Mapped[str] = mapped_column(project_task_t)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()


class Dataset(Base):
    """1:1 with a project: the dataset id IS the project id."""

    __tablename__ = "datasets"

    project_id: Mapped[uuid.UUID] = _fk("projects.id", primary_key=True)
    modality: Mapped[str] = mapped_column(modality_t)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        sa.UniqueConstraint("dataset_id", "version_tag", name="datasetVersions_datasetId_versionTag_key"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = _fk("datasets.project_id", index=True)
    # NULL means the project draft; NOT NULL means an immutable snapshot used for training.
    version_tag: Mapped[str | None] = mapped_column(sa.String(50))
    status: Mapped[str] = mapped_column(dataset_version_status_t, server_default="draft")
    item_count: Mapped[int | None] = mapped_column(sa.Integer)
    class_count: Mapped[int | None] = mapped_column(sa.Integer)
    parquet_key: Mapped[str | None] = mapped_column(sa.Text)
    # What the snapshot was built with (theseus.augmentation.config.AugmentationConfig), NULL for none.
    augmentation_config: Mapped[Any | None] = mapped_column(JSONB)
    # Augmented copies added to the train split; included in item_count.
    augmented_count: Mapped[int] = mapped_column(sa.Integer, server_default="0")
    failed_message: Mapped[str | None] = mapped_column(sa.Text)
    built_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at()


class DatasetItem(Base):
    """A row in the project-wide, content-addressed, deduplicated pool.

    Augmented copies are rows too, but they are NOT pool members: `source_item_id` marks them, they
    belong to exactly one snapshot (never the draft), and their files live under that snapshot's
    S3 prefix. Pool dedup on (dataset_id, content_hash) therefore only covers originals.
    """

    __tablename__ = "dataset_items"
    __table_args__ = (
        sa.Index(
            "dataset_items_pool_contentHash_key",
            "dataset_id",
            "content_hash",
            unique=True,
            postgresql_where=sa.text("source_item_id IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = _fk("datasets.project_id", index=True)
    external_id: Mapped[str | None] = mapped_column(sa.String(255))
    storage_url: Mapped[str | None] = mapped_column(sa.Text)
    content_hash: Mapped[str | None] = mapped_column(sa.CHAR(64))
    byte_size: Mapped[int | None] = mapped_column(sa.Integer)
    # The original this item was augmented from; NULL for real (pool) items. RESTRICT: an original
    # with augmented copies is soft-deleted, never hard-deleted, like one referenced by a snapshot.
    source_item_id: Mapped[uuid.UUID | None] = mapped_column(
        sa.ForeignKey("dataset_items.id", ondelete="RESTRICT"), index=True
    )
    # The ops (and params) that produced this copy.
    augmentation: Mapped[Any | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_at()
    # Set instead of hard-deleting when a snapshot RESTRICT FK blocks the delete.
    deleted_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))


class DatasetVersionItem(Base):
    __tablename__ = "dataset_version_items"
    __table_args__ = (sa.Index("datasetVersionItems_versionId_splitType_idx", "version_id", "split_type"),)

    version_id: Mapped[uuid.UUID] = _fk("dataset_versions.id", primary_key=True)
    # RESTRICT: an item referenced by any snapshot cannot be hard-deleted, which is what
    # makes a snapshot genuinely immutable.
    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", ondelete="RESTRICT", primary_key=True)
    split_type: Mapped[str] = mapped_column(split_type_t)


class TextFeatures(Base):
    __tablename__ = "text_features"

    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", primary_key=True)
    raw_text: Mapped[str] = mapped_column(sa.Text)
    token_count: Mapped[int | None] = mapped_column(sa.Integer)
    language_code: Mapped[str | None] = mapped_column(sa.String(10), index=True)
    meta_json: Mapped[Any | None] = mapped_column(JSONB)


class VisionFeatures(Base):
    __tablename__ = "vision_features"

    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", primary_key=True)
    width: Mapped[int] = mapped_column(sa.Integer)
    height: Mapped[int] = mapped_column(sa.Integer)
    channels: Mapped[int | None] = mapped_column(sa.Integer, server_default="3")
    image_format: Mapped[str | None] = mapped_column(image_format_t)
    exif_data: Mapped[Any | None] = mapped_column(JSONB)


class AudioFeatures(Base):
    __tablename__ = "audio_features"

    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", primary_key=True)
    duration_seconds: Mapped[Decimal] = mapped_column(sa.Numeric(8, 3))
    sample_rate_hz: Mapped[int] = mapped_column(sa.Integer)
    channels: Mapped[int | None] = mapped_column(sa.Integer, server_default="1")
    audio_codec: Mapped[str | None] = mapped_column(audio_codec_t)


class TabularFeatures(Base):
    __tablename__ = "tabular_features"
    __table_args__ = (sa.Index("idx_tabular_features_gin", "features_json", postgresql_using="gin"),)

    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", primary_key=True)
    features_json: Mapped[Any] = mapped_column(JSONB)


class LabelClass(Base):
    __tablename__ = "label_classes"
    __table_args__ = (sa.UniqueConstraint("dataset_id", "name", name="labelClasses_datasetId_name_key"),)

    class_id: Mapped[uuid.UUID] = uuid_pk()
    dataset_id: Mapped[uuid.UUID] = _fk("datasets.project_id", index=True)
    name: Mapped[str] = mapped_column(sa.String(100))
    description: Mapped[str | None] = mapped_column(sa.Text)
    ui_color_hex: Mapped[str | None] = mapped_column(sa.String(7), server_default="#FFFFFF")
    is_active: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.true())
    created_at: Mapped[datetime | None] = created_at()


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        sa.Index("idx_annotations_class", "class_id", "annotation_type"),
        # At most one classification label per item. Without it, two concurrent classify calls
        # both see no existing label and both insert, and the snapshot builder then bakes a
        # nondeterministic ground-truth label into the parquet.
        sa.Index(
            "annotations_item_classification_key",
            "item_id",
            unique=True,
            postgresql_where=sa.text("annotation_type = 'classification'"),
        ),
        sa.CheckConstraint("confidence_score BETWEEN 0.0 AND 1.0", name="confidence_bounds"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    item_id: Mapped[uuid.UUID] = _fk("dataset_items.id", index=True)
    annotator_id: Mapped[str | None] = mapped_column(sa.String(100))
    annotation_type: Mapped[str] = mapped_column(annotation_type_t, index=True)
    class_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("label_classes.class_id", ondelete="RESTRICT"))
    label_text_sequence: Mapped[str | None] = mapped_column(sa.Text)
    label_structured: Mapped[Any | None] = mapped_column(JSONB)
    confidence_score: Mapped[Decimal | None] = mapped_column(sa.Numeric(4, 3))
    created_at: Mapped[datetime | None] = created_at()
