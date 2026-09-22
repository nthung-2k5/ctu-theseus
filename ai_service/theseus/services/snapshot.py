"""Build a version dataset.parquet and manifest from its pool-item membership.

Ported from server/lib/snapshot.ts (hyparquet-writer replaced by pyarrow). The parquet holds S3
URIs and label/scalar values, not raw bytes: files stay in the content-addressed pool and
Ludwig reads them through s3fs, so a snapshot is metadata-sized and cheap to build inline.
"""

import asyncio
import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa

from theseus import constants as C
from theseus.db.base import get_sessionmaker
from theseus.db.models import (
    Annotation,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    TabularFeatures,
    TextFeatures,
)
from theseus.services import storage
from theseus.services.augmentation import delete_augmented_items, materialize_augmentations
from theseus.services.task_registry import ColumnSpec, SnapshotContext, TaskDescriptor, get_task_descriptor

logger = logging.getLogger(__name__)

# Ludwig fixed split preprocessing needs an integer column: its splitter does
# column.astype(np.int8) and partitions on the literal values 0/1/2 (train/validation/test), so
# the human-readable `split` string column cannot double as it. This one is synthetic.
SPLIT_INDEX = {"train": 0, "validation": 1, "test": 2}


@dataclass
class AnnotationView:
    class_id: uuid.UUID | None
    annotation_type: str
    label_structured: Any = None
    label_text_sequence: str | None = None


@dataclass
class SnapshotMember:
    """One version item with everything a column resolver may need."""

    item_id: uuid.UUID
    split_type: str
    storage_url: str | None = None
    raw_text: str | None = None
    features_json: dict[str, Any] | None = None
    annotations: list[AnnotationView] = field(default_factory=list)


def resolve_column_value(col: ColumnSpec, member: SnapshotMember, class_name_by_id: dict[uuid.UUID, str]) -> Any:
    match col.kind:
        case "storage_uri":
            return f"s3://{C.BUCKET_DATASETS}/{member.storage_url}" if member.storage_url else None
        case "inline_text":
            return member.raw_text
        case "split":
            return member.split_type
        case "split_index":
            return SPLIT_INDEX[member.split_type]
        case "item_id":
            return str(member.item_id)
        case "label":
            classification = next((a for a in member.annotations if a.class_id is not None), None)
            if classification is not None and classification.class_id is not None:
                return class_name_by_id.get(classification.class_id)
            # Regression targets are stored as {value: number} in label_structured, since
            # annotations were designed classification-first.
            structured = member.annotations[0].label_structured if member.annotations else None
            return structured.get("value") if isinstance(structured, dict) else None
        case "scalar":
            return (member.features_json or {}).get(col.name)
        case "text_sequence_label":
            seq = next((a for a in member.annotations if a.annotation_type == "text_sequence"), None)
            return seq.label_text_sequence if seq else None
    raise ValueError(f"Unknown column kind: {col.kind}")


def _arrow_type(col: ColumnSpec, task: TaskDescriptor) -> pa.DataType:
    if col.kind == "scalar":
        return pa.float64()
    if col.kind == "split_index":
        return pa.int32()
    # A label column normally holds a class name, but a regression target holds a number.
    if col.kind == "label" and task.ludwig is not None:
        if any(f["column"] == col.name and f["type"] == "number" for f in task.ludwig.output_features):
            return pa.float64()
    return pa.string()


def _coerce(value: Any, typ: pa.DataType) -> Any:
    if value is None:
        return None
    if pa.types.is_floating(typ):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None  # a non-numeric cell in a numeric column becomes a null, not a failed build
    return value


def derive_columns(task: TaskDescriptor, members: list[SnapshotMember]) -> list[ColumnSpec]:
    """The task columns, plus dataset-defined scalar columns for tabular tasks, plus the two synthetic ones."""
    scalar: list[ColumnSpec] = []
    if task.modality == "tabular" and not any(c.kind == "scalar" for c in task.columns):
        # Tabular feature columns are dataset-defined: take the union of keys in first-seen order.
        seen: dict[str, None] = {}
        for m in members:
            for key in m.features_json or {}:
                seen.setdefault(key, None)
        scalar = [ColumnSpec(name=k, kind="scalar") for k in seen]
    return [
        *task.columns,
        *scalar,
        ColumnSpec(name=C.SPLIT_INDEX_COLUMN_NAME, kind="split_index"),
        ColumnSpec(name=C.ITEM_ID_COLUMN_NAME, kind="item_id"),
    ]


def build_rows(
    columns: list[ColumnSpec], members: list[SnapshotMember], class_name_by_id: dict[uuid.UUID, str]
) -> list[dict[str, Any]]:
    return [{c.name: resolve_column_value(c, m, class_name_by_id) for c in columns} for m in members]


def class_counts(task: TaskDescriptor, columns: list[ColumnSpec], rows: list[dict[str, Any]]) -> dict[str, int] | None:
    """Class distribution, classification tasks only (a regression target is also kind=label but numeric)."""
    label = next((c for c in columns if c.kind == "label"), None)
    if label is None or not task.annotation.requires_label_classes:
        return None
    counts: dict[str, int] = {}
    for row in rows:
        value = row[label.name]
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
    return counts


def to_parquet_bytes(task: TaskDescriptor, columns: list[ColumnSpec], rows: list[dict[str, Any]]) -> bytes:
    types = {c.name: _arrow_type(c, task) for c in columns}
    arrays = [pa.array([_coerce(r[c.name], types[c.name]) for r in rows], type=types[c.name]) for c in columns]
    table = pa.Table.from_arrays(arrays, schema=pa.schema([(c.name, types[c.name]) for c in columns]))
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def build_manifest(
    members: list[SnapshotMember], class_names: list[str], counts: dict[str, int] | None, columns: list[ColumnSpec]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "itemCount": len(members),
        "classCount": len(class_names),
        "classes": class_names,
        "columns": [c.model_dump(by_alias=True) for c in columns],
        "createdAt": datetime.now(UTC).isoformat(),
    }
    if counts is not None:
        manifest["classCounts"] = counts
    return manifest


# -- Orchestration (DB + S3) -----------------------------------------------------------------


async def _load_members(session, version_id: uuid.UUID) -> list[SnapshotMember]:
    rows = (
        await session.execute(
            sa.select(DatasetVersionItem.item_id, DatasetVersionItem.split_type, DatasetItem.storage_url)
            .join(DatasetItem, DatasetItem.id == DatasetVersionItem.item_id)
            .where(DatasetVersionItem.version_id == version_id)
            .order_by(DatasetItem.id)
        )
    ).all()
    members = {r.item_id: SnapshotMember(r.item_id, r.split_type, r.storage_url) for r in rows}
    in_version = sa.select(DatasetVersionItem.item_id).where(DatasetVersionItem.version_id == version_id)

    for item_id, raw_text in (
        await session.execute(
            sa.select(TextFeatures.item_id, TextFeatures.raw_text).where(TextFeatures.item_id.in_(in_version))
        )
    ).all():
        members[item_id].raw_text = raw_text
    for item_id, features in (
        await session.execute(
            sa.select(TabularFeatures.item_id, TabularFeatures.features_json).where(
                TabularFeatures.item_id.in_(in_version)
            )
        )
    ).all():
        members[item_id].features_json = features
    for a in (
        await session.execute(
            sa.select(Annotation)
            .where(Annotation.item_id.in_(in_version))
            .order_by(Annotation.created_at, Annotation.id)
        )
    ).scalars():
        members[a.item_id].annotations.append(
            AnnotationView(a.class_id, a.annotation_type, a.label_structured, a.label_text_sequence)
        )
    return list(members.values())


async def build_snapshot(version_id: uuid.UUID) -> None:
    """Build the parquet and manifest, then flip building -> ready | failed. Never raises."""
    sessionmaker = get_sessionmaker()
    loop = asyncio.get_running_loop()
    augmentation_config: dict[str, Any] | None = None
    augmented = 0
    try:
        # Phase 1 (only when requested): materialize augmented train-split copies as real items, so the
        # membership load below, and with it the parquet, manifest and counts, includes them unchanged.
        async with sessionmaker() as session:
            version = await session.get(DatasetVersion, version_id)
            if version is None:
                raise ValueError(f"Version {version_id} not found")
            augmentation_config = version.augmentation_config
        if augmentation_config:
            result = await materialize_augmentations(version_id)
            augmented = result.created
            if result.skipped:
                augmentation_config = {**augmentation_config, "skippedItems": result.skipped}

        async with sessionmaker() as session:
            version = await session.get(DatasetVersion, version_id)
            if version is None:
                raise ValueError(f"Version {version_id} not found")
            project = await session.get(Project, version.dataset_id)
            if project is None:
                raise ValueError(f"Version {version_id} project not found")
            task = get_task_descriptor(project.task)
            members = await _load_members(session, version_id)
            class_rows = (
                (
                    await session.execute(
                        sa.select(LabelClass).where(
                            LabelClass.dataset_id == version.dataset_id, LabelClass.is_active.is_(True)
                        )
                    )
                )
                .scalars()
                .all()
            )
            class_name_by_id = {c.class_id: c.name for c in class_rows}
            class_names = [c.name for c in class_rows]

        columns = derive_columns(task, members)
        rows = build_rows(columns, members, class_name_by_id)
        counts = class_counts(task, columns, rows)
        manifest = build_manifest(members, class_names, counts, columns)

        parquet_key = storage.snapshot_parquet_key(str(version_id))

        def _write() -> None:
            data = to_parquet_bytes(task, columns, rows)
            storage.upload_bytes(C.BUCKET_DATASETS, parquet_key, data, "application/octet-stream")
            storage.upload_bytes(
                C.BUCKET_DATASETS,
                storage.snapshot_manifest_key(str(version_id)),
                json.dumps(manifest, indent=2).encode(),
                "application/json",
            )

        await loop.run_in_executor(None, _write)

        async with sessionmaker() as session:
            await session.execute(
                sa.update(DatasetVersion)
                .where(DatasetVersion.id == version_id, DatasetVersion.status == "building")
                .values(
                    status="ready",
                    item_count=len(members),
                    class_count=len(class_names),
                    parquet_key=parquet_key,
                    augmented_count=augmented,
                    augmentation_config=augmentation_config,
                    built_at=sa.func.now(),
                )
            )
            await session.commit()
    except Exception as e:
        logger.exception("Snapshot build failed for version %s", version_id)
        try:
            async with sessionmaker() as session:
                await session.execute(
                    sa.update(DatasetVersion)
                    .where(DatasetVersion.id == version_id, DatasetVersion.status == "building")
                    .values(status="failed", failed_message=str(e))
                )
                await session.commit()
        except Exception:
            # Startup recovery fails any version left in `building`, so this is not fatal.
            logger.exception("Could not record snapshot failure for version %s", version_id)
        if augmentation_config:
            await _discard_partial_augmentation(version_id)


async def _discard_partial_augmentation(version_id: uuid.UUID) -> None:
    """A failed build must not leave half an augmentation behind: its rows and S3 objects are useless
    (nothing trains on a failed snapshot) and would otherwise linger until the version is deleted."""
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, storage.delete_prefix, C.BUCKET_DATASETS, storage.augmented_prefix(str(version_id))
        )
        async with get_sessionmaker()() as session:
            await delete_augmented_items(session, version_id)
            await session.commit()
    except Exception:
        logger.exception("Could not discard partial augmentation for version %s", version_id)


async def read_snapshot_manifest(version_id: uuid.UUID | str) -> SnapshotContext:
    """Rebuild the SnapshotContext for the Ludwig compiler from a stored manifest."""
    raw = await asyncio.get_running_loop().run_in_executor(
        None, storage.download_bytes, C.BUCKET_DATASETS, storage.snapshot_manifest_key(str(version_id))
    )
    manifest = json.loads(raw)
    return SnapshotContext(
        columns=[ColumnSpec(**c) for c in manifest["columns"]],
        label_class_names=manifest["classes"],
        class_counts=manifest.get("classCounts"),
    )
