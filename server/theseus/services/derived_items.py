"""Shared machinery between the two snapshot-time item transforms: preprocessing (services/preprocessing.py)
and augmentation (services/augmentation.py). Both read pool-ish items out of a version's membership,
decode/encode the same file-backed modalities, and write the same feature-row shape for a new item.
"""

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.db.models import (
    Annotation,
    AudioFeatures,
    DatasetItem,
    DatasetVersionItem,
    TabularFeatures,
    TextFeatures,
    VisionFeatures,
)

_DELETE_CHUNK = 5000


@dataclass
class Original:
    id: uuid.UUID
    split_type: str
    external_id: str | None
    storage_url: str | None
    content_hash: str | None
    # The pool ancestor, for an item that is itself a preprocessed replacement; None for a real pool item.
    source_item_id: uuid.UUID | None = None
    # The preprocessing ops (if any) already applied to this item, carried forward onto any copy made from it.
    preprocessing: Any | None = None
    text: dict[str, Any] | None = None
    features_json: Any = None
    annotations: list[Annotation] = field(default_factory=list)


async def load_items(
    session: AsyncSession, version_id: uuid.UUID, modality: str, *, splits: set[str], where: Any
) -> list[Original]:
    """A version's items in the given splits (plus `where`, an extra DatasetItem-level filter) with
    what each modality needs to preprocess or augment it."""
    rows = (
        await session.execute(
            sa.select(DatasetItem, DatasetVersionItem.split_type)
            .join(DatasetVersionItem, DatasetVersionItem.item_id == DatasetItem.id)
            .where(DatasetVersionItem.version_id == version_id, DatasetVersionItem.split_type.in_(splits), where)
            .order_by(DatasetItem.id)
        )
    ).all()
    originals = {
        i.id: Original(
            i.id,
            split,
            i.external_id,
            i.storage_url,
            i.content_hash,
            source_item_id=i.source_item_id,
            preprocessing=i.preprocessing,
        )  # fmt: skip
        for i, split in rows
    }
    if not originals:
        return []
    ids = list(originals)

    if modality == "text":
        for f in (await session.execute(sa.select(TextFeatures).where(TextFeatures.item_id.in_(ids)))).scalars():
            originals[f.item_id].text = {
                "raw_text": f.raw_text,
                "language_code": f.language_code,
                "meta_json": f.meta_json,
            }
    elif modality == "tabular":
        for f in (await session.execute(sa.select(TabularFeatures).where(TabularFeatures.item_id.in_(ids)))).scalars():
            originals[f.item_id].features_json = f.features_json

    for a in (
        await session.execute(
            sa.select(Annotation).where(Annotation.item_id.in_(ids)).order_by(Annotation.created_at, Annotation.id)
        )
    ).scalars():
        originals[a.item_id].annotations.append(a)
    return list(originals.values())


def has_content(original: Original, modality: str) -> bool:
    if modality == "text":
        return original.text is not None
    if modality == "tabular":
        return isinstance(original.features_json, dict)
    return original.storage_url is not None


def samples_for_fit(originals: list[Original], modality: str) -> list[Any]:
    """Inline samples (text/tabular) a plugin's `fit`/`prepare` can use; file-backed modalities get none."""
    if modality == "text":
        return [o.text["raw_text"] for o in originals if o.text]
    if modality == "tabular":
        return [o.features_json for o in originals if isinstance(o.features_json, dict)]
    return []


def feature_row(modality: str, item_id: uuid.UUID, f: dict[str, Any]) -> Any:
    if modality == "vision":
        return VisionFeatures(
            item_id=item_id,
            width=f["width"],
            height=f["height"],
            channels=f["channels"],
            image_format=f["image_format"],
        )
    if modality == "audio":
        return AudioFeatures(
            item_id=item_id,
            duration_seconds=Decimal(str(f["duration_seconds"])),
            sample_rate_hz=f["sample_rate_hz"],
            channels=f["channels"],
            audio_codec=f["audio_codec"],
        )
    if modality == "text":
        return TextFeatures(
            item_id=item_id,
            raw_text=f["raw_text"],
            token_count=f["token_count"],
            language_code=f["language_code"],
            meta_json=f["meta_json"],
        )
    return TabularFeatures(item_id=item_id, features_json=f["features_json"])


async def delete_derived_items(session: AsyncSession, version_id: uuid.UUID) -> int:
    """Delete a snapshot's derived items — preprocessed replacements and augmented copies alike
    (rows only; S3 is cleanup_version_storage's job).

    Membership rows go first: dataset_version_items.item_id is ON DELETE RESTRICT. Features and
    annotations cascade from the item row.
    """
    ids = (
        (
            await session.execute(
                sa.select(DatasetItem.id)
                .join(DatasetVersionItem, DatasetVersionItem.item_id == DatasetItem.id)
                .where(DatasetVersionItem.version_id == version_id, DatasetItem.source_item_id.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    for start in range(0, len(ids), _DELETE_CHUNK):
        chunk = ids[start : start + _DELETE_CHUNK]
        await session.execute(
            sa.delete(DatasetVersionItem).where(
                DatasetVersionItem.version_id == version_id, DatasetVersionItem.item_id.in_(chunk)
            )
        )
        await session.execute(sa.delete(DatasetItem).where(DatasetItem.id.in_(chunk)))
    return len(ids)


def copy_annotations(session: AsyncSession, item_id: uuid.UUID, annotations: list[Annotation]) -> None:
    """Duplicate an original's annotations onto a new derived item."""
    for a in annotations:
        session.add(
            Annotation(
                item_id=item_id,
                annotator_id=a.annotator_id,
                annotation_type=a.annotation_type,
                class_id=a.class_id,
                label_text_sequence=a.label_text_sequence,
                label_structured=a.label_structured,
                confidence_score=a.confidence_score,
            )
        )
