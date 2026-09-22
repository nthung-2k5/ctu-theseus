"""Dataset pool logic: dedup hashing, item creation, deletion, and split planning.

Ported from the helpers and inline logic of server/routes/datasets.ts.
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.db.models import (
    Annotation,
    AudioFeatures,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    TabularFeatures,
    TextFeatures,
    VisionFeatures,
)
from theseus.schemas.datasets import ItemIn
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)

# A class with fewer than this many items cannot appear in all three splits (auto-split) and cannot
# be meaningfully evaluated on its own (health check). One shared threshold so the two cannot drift.
MIN_ITEMS_PER_CLASS = 3

SPLITS: tuple[str, ...] = ("train", "validation", "test")


def js_round(x: float) -> int:
    """JavaScript Math.round (halves round up). Python round() is banker's rounding and would
    change split sizes relative to the gateway for exact halves."""
    return math.floor(x + 0.5)


# -- Content hashing (pool dedup for inline items) -------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON with sorted keys, so object key order cannot change an item hash."""
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        entries = sorted(value.items(), key=lambda kv: kv[0])
        return "{" + ",".join(f"{json.dumps(k)}:{canonical_json(v)}" for k, v in entries) + "}"
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def hash_item_content(item: ItemIn) -> str | None:
    """sha256 of an inline (text/tabular) item, for the same (dataset, content_hash) dedup file uploads use."""
    if item.text_features is not None:
        return hashlib.sha256(item.text_features.raw_text.encode()).hexdigest()
    if item.tabular_features is not None:
        return hashlib.sha256(canonical_json(item.tabular_features.features_json).encode()).hexdigest()
    return None


def _int_or_none(v: float | None) -> int | None:
    return None if v is None else int(v)


# -- Guards ----------------------------------------------------------------------------------


NO_LABEL_CLASSES = "This task does not use label classes"


def task_uses_label_classes(task: str) -> bool:
    """Whether a project's task labels items with classes. Regression and free-text tasks (captioning,
    ASR, generation) do not, so a class must never be attached to their items."""
    return get_task_descriptor(task).annotation.requires_label_classes


async def class_in_dataset(session: AsyncSession, class_id: uuid.UUID, dataset_id: uuid.UUID) -> bool:
    """Guard against attaching another tenant label class to an item.

    The FK on annotations.class_id only proves the class exists somewhere: a class from a different
    dataset passes it, then resolves to null in the snapshot builder dataset-scoped name map,
    silently producing unlabeled training rows.
    """
    found = (
        await session.execute(
            sa.select(LabelClass.class_id).where(LabelClass.class_id == class_id, LabelClass.dataset_id == dataset_id)
        )
    ).first()
    return found is not None


# -- Item creation ---------------------------------------------------------------------------


async def create_item(session: AsyncSession, dataset_id: uuid.UUID, draft_id: uuid.UUID, data: ItemIn) -> DatasetItem:
    """Create one item with its features, annotations and draft membership as one unit.

    The caller wraps this in a savepoint so a partial failure (say the item row commits but its
    features row does not) never leaves an orphaned item, while one bad item still cannot fail the
    whole batch.
    """
    content_hash = hash_item_content(data)
    existing = None
    if content_hash:
        existing = (
            await session.execute(
                sa.select(DatasetItem).where(
                    DatasetItem.dataset_id == dataset_id,
                    DatasetItem.content_hash == content_hash,
                    DatasetItem.source_item_id.is_(None),  # never dedup onto an augmented copy
                )
            )
        ).scalar_one_or_none()

    if existing is not None:
        item = existing
        # Re-adding a soft-deleted item exact content restores it (mirrors the upload route).
        if item.deleted_at is not None:
            item.deleted_at = None
    else:
        item = DatasetItem(dataset_id=dataset_id, external_id=data.external_id, content_hash=content_hash)
        session.add(item)
        await session.flush()
        # Only a freshly created item gets features and annotations. A dedup hit means the content is
        # already in the pool, possibly already labeled differently, and the feature tables are 1:1.
        if data.text_features:
            f = data.text_features
            session.add(
                TextFeatures(
                    item_id=item.id, raw_text=f.raw_text, token_count=_int_or_none(f.token_count),
                    language_code=f.language_code, meta_json=f.meta_json,
                )
            )  # fmt: skip
        if data.vision_features:
            f = data.vision_features
            session.add(
                VisionFeatures(
                    item_id=item.id, width=int(f.width), height=int(f.height), channels=_int_or_none(f.channels),
                    image_format=f.image_format, exif_data=f.exif_data,
                )
            )  # fmt: skip
        if data.audio_features:
            f = data.audio_features
            session.add(
                AudioFeatures(
                    item_id=item.id, duration_seconds=Decimal(str(f.duration_seconds)),
                    sample_rate_hz=int(f.sample_rate_hz), channels=_int_or_none(f.channels), audio_codec=f.audio_codec,
                )
            )  # fmt: skip
        if data.tabular_features:
            session.add(TabularFeatures(item_id=item.id, features_json=data.tabular_features.features_json))
        for ann in data.annotations or []:
            session.add(
                Annotation(
                    item_id=item.id, annotator_id=ann.annotator_id, annotation_type=ann.annotation_type,
                    class_id=ann.class_id, label_text_sequence=ann.label_text_sequence,
                    label_structured=ann.label_structured,
                    confidence_score=None if ann.confidence_score is None else Decimal(str(ann.confidence_score)),
                )
            )  # fmt: skip
        await session.flush()

    await session.execute(
        pg_insert(DatasetVersionItem)
        .values(version_id=draft_id, item_id=item.id, split_type=data.split)
        .on_conflict_do_nothing()
    )
    return item


def safe_item_error(exc: BaseException) -> str:
    """A message safe to hand an API caller: raw driver text can embed constraint, table and column names."""
    if isinstance(exc, IntegrityError) and getattr(exc.orig, "sqlstate", None) == "23505":
        return "An item with identical content already exists in this project"
    return "Failed to create item"


# -- Deletion --------------------------------------------------------------------------------


async def delete_item_from_pool(session: AsyncSession, item_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    """Remove an item from the draft, then try a hard delete: 'deleted' | 'soft_deleted' | 'not_found'.

    If a SNAPSHOT still references it, the RESTRICT foreign key on dataset_version_items blocks the
    hard delete. Soft-delete then: the row (and its features and annotations) stays intact for that
    snapshot while it disappears from the pool, which the draft-membership removal already achieves.
    """
    draft_id = (
        await session.execute(
            sa.select(DatasetVersion.id).where(
                DatasetVersion.dataset_id == dataset_id, DatasetVersion.version_tag.is_(None)
            )
        )
    ).scalar_one_or_none()
    if draft_id is not None:
        await session.execute(
            sa.delete(DatasetVersionItem).where(
                DatasetVersionItem.version_id == draft_id, DatasetVersionItem.item_id == item_id
            )
        )

    try:
        async with session.begin_nested():
            storage_url = (
                await session.execute(
                    sa.delete(DatasetItem)
                    .where(DatasetItem.id == item_id, DatasetItem.dataset_id == dataset_id)
                    .returning(DatasetItem.storage_url)
                )
            ).first()
    except IntegrityError as e:
        # 23503 foreign_key_violation (what ON DELETE RESTRICT actually raises) or 23001 restrict_violation.
        if getattr(e.orig, "sqlstate", None) not in ("23503", "23001"):
            raise
        soft = (
            await session.execute(
                sa.update(DatasetItem)
                .where(DatasetItem.id == item_id, DatasetItem.dataset_id == dataset_id)
                .values(deleted_at=sa.func.now())
                .returning(DatasetItem.id)
            )
        ).first()
        return "soft_deleted" if soft else "not_found"

    if storage_url is None:
        return "not_found"
    if storage_url[0]:
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, storage.delete_file, C.BUCKET_DATASETS, storage_url[0]
            )
        except Exception:
            logger.warning("Could not delete pool object %s", storage_url[0], exc_info=True)
    return "deleted"


# -- Auto-split planning (pure, so it can be tested without a database) ----------------------


@dataclass
class SplitPlan:
    groups: dict[str, list[uuid.UUID]]
    warnings: list[str]


def plan_split(
    members: Sequence[tuple[uuid.UUID, str]],
    ratios: dict[str, float],
    stratify: bool,
    rng: random.Random | None = None,
) -> SplitPlan:
    """Assign every member to exactly one split by ratio, optionally stratified by label class.

    members is (item_id, class_key) where class_key is the class id or "unassigned". Un-stratified,
    everything is one group. Stratified, each class is shuffled and cut on its own, so a class
    present in the draft cannot be shuffled entirely out of validation or test.
    """
    rng = rng or random.Random()
    total = ratios["train"] + ratios["validation"] + ratios["test"]
    if total <= 0:
        raise ValueError("Ratios must sum to a positive number")

    by_key: dict[str, list[uuid.UUID]] = {}
    for item_id, key in members:
        by_key.setdefault(key if stratify else "all", []).append(item_id)

    groups: dict[str, list[uuid.UUID]] = {s: [] for s in SPLITS}
    warnings: list[str] = []
    for key, ids in by_key.items():
        rng.shuffle(ids)
        train_n = js_round(ratios["train"] / total * len(ids))
        val_n = js_round(ratios["validation"] / total * len(ids))
        # test absorbs the rounding remainder so every item is assigned exactly once.
        groups["train"].extend(ids[:train_n])
        groups["validation"].extend(ids[train_n : train_n + val_n])
        groups["test"].extend(ids[train_n + val_n :])
        if stratify and key != "unassigned" and len(ids) < MIN_ITEMS_PER_CLASS:
            warnings.append(f"Class {key} has only {len(ids)} item(s), so it cannot appear in all three splits.")
    return SplitPlan(groups, warnings)


# -- Bulk creation (fast path) ---------------------------------------------------------------

_CHUNK = 5000


async def create_items_bulk(
    session: AsyncSession, dataset_id: uuid.UUID, draft_id: uuid.UUID, items: Sequence[ItemIn]
) -> list[DatasetItem]:
    """Create a whole batch with batched inserts instead of one round trip per row.

    Same semantics as create_item: content-addressed dedup (within the batch and against the pool),
    features and annotations only for freshly created items, soft-deleted matches restored, and
    membership ON CONFLICT DO NOTHING so the first split wins. Returns one DatasetItem per input, in
    order. All-or-nothing: the caller runs it in a savepoint and falls back to per-item creation
    when any row violates a constraint, so one bad row still fails alone.
    """
    hashes = [hash_item_content(i) for i in items]
    wanted = list({h for h in hashes if h})
    known: dict[str, DatasetItem] = {}
    for start in range(0, len(wanted), _CHUNK):
        rows = await session.execute(
            sa.select(DatasetItem).where(
                DatasetItem.dataset_id == dataset_id,
                DatasetItem.content_hash.in_(wanted[start : start + _CHUNK]),
                DatasetItem.source_item_id.is_(None),  # never dedup onto an augmented copy
            )
        )
        known.update({r.content_hash: r for r in rows.scalars()})

    result: list[DatasetItem] = []
    fresh: list[tuple[ItemIn, DatasetItem]] = []
    restored = False
    for data, h in zip(items, hashes, strict=True):
        if h and h in known:
            item = known[h]
            if item.deleted_at is not None:
                item.deleted_at = None
                restored = True
        else:
            item = DatasetItem(dataset_id=dataset_id, external_id=data.external_id, content_hash=h)
            fresh.append((data, item))
            if h:
                known[h] = item  # a duplicate later in this batch resolves to the same row
        result.append(item)

    session.add_all([item for _, item in fresh])
    if fresh or restored:
        await session.flush()

    extras: list[Any] = []
    for data, item in fresh:
        if data.text_features:
            f = data.text_features
            extras.append(
                TextFeatures(
                    item_id=item.id, raw_text=f.raw_text, token_count=_int_or_none(f.token_count),
                    language_code=f.language_code, meta_json=f.meta_json,
                )
            )  # fmt: skip
        if data.vision_features:
            f = data.vision_features
            extras.append(
                VisionFeatures(
                    item_id=item.id, width=int(f.width), height=int(f.height), channels=_int_or_none(f.channels),
                    image_format=f.image_format, exif_data=f.exif_data,
                )
            )  # fmt: skip
        if data.audio_features:
            f = data.audio_features
            extras.append(
                AudioFeatures(
                    item_id=item.id, duration_seconds=Decimal(str(f.duration_seconds)),
                    sample_rate_hz=int(f.sample_rate_hz), channels=_int_or_none(f.channels), audio_codec=f.audio_codec,
                )
            )  # fmt: skip
        if data.tabular_features:
            extras.append(TabularFeatures(item_id=item.id, features_json=data.tabular_features.features_json))
        for ann in data.annotations or []:
            extras.append(
                Annotation(
                    item_id=item.id, annotator_id=ann.annotator_id, annotation_type=ann.annotation_type,
                    class_id=ann.class_id, label_text_sequence=ann.label_text_sequence,
                    label_structured=ann.label_structured,
                    confidence_score=None if ann.confidence_score is None else Decimal(str(ann.confidence_score)),
                )
            )  # fmt: skip
    session.add_all(extras)
    await session.flush()

    membership = [
        {"version_id": draft_id, "item_id": item.id, "split_type": data.split}
        for data, item in zip(items, result, strict=True)
    ]
    for start in range(0, len(membership), _CHUNK):
        await session.execute(
            pg_insert(DatasetVersionItem).values(membership[start : start + _CHUNK]).on_conflict_do_nothing()
        )
    return result
