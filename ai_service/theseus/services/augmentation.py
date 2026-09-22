"""Materialize a snapshot's augmented items: the TRAIN split gets N extra, real, browsable copies.

Runs as the first phase of build_snapshot. Each copy is a dataset_items row (with `source_item_id`
pointing at its original, plus the modality features and a copy of the original's annotations) and
a dataset_version_items row in the train split, so the parquet, manifest, split counts and the
items browser all pick it up with no special casing. File-backed copies are written under the
snapshot's own S3 prefix, never the content-addressed pool (see storage.augmented_prefix).

The work per original (download, decode, augment, encode, upload) is blocking and CPU-heavy, so it
runs in worker threads with bounded concurrency; database writes happen on the event loop, one
transaction per batch. Copies are seeded from (snapshot, item, copy index), so a rebuild reproduces
the same augmented items.
"""

import asyncio
import hashlib
import logging
import random
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
import uuid_utils
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.augmentation import media
from theseus.augmentation.config import AugmentationConfig
from theseus.augmentation.pipeline import ConfiguredOp, augment, build_plan, seed_for
from theseus.db.base import get_sessionmaker
from theseus.db.models import (
    Annotation,
    AudioFeatures,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    Project,
    TabularFeatures,
    TextFeatures,
    VisionFeatures,
)
from theseus.services import storage
from theseus.services.datasets import canonical_json
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)

_BATCH = 16  # originals per DB transaction
_CONCURRENCY = 4  # originals augmented at once (threads)
_DELETE_CHUNK = 5000


@dataclass
class Original:
    id: uuid.UUID
    external_id: str | None
    storage_url: str | None
    content_hash: str | None
    text: dict[str, Any] | None = None
    features_json: Any = None
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class Produced:
    copy_number: int  # 1-based
    external_id: str
    storage_url: str | None
    content_hash: str
    byte_size: int | None
    # Modality feature-row columns for the new item.
    features: dict[str, Any]
    applied: list[dict[str, Any]]


@dataclass
class AugmentResult:
    created: int = 0
    skipped: int = 0  # originals that could not be augmented
    total: int = 0  # originals attempted


# -- Loading ---------------------------------------------------------------------------------


async def _load_originals(session: AsyncSession, version_id: uuid.UUID, modality: str) -> list[Original]:
    """The version's train-split originals (never augmented copies) with what each modality needs."""
    rows = (
        await session.execute(
            sa.select(DatasetItem)
            .join(DatasetVersionItem, DatasetVersionItem.item_id == DatasetItem.id)
            .where(
                DatasetVersionItem.version_id == version_id,
                DatasetVersionItem.split_type == "train",
                DatasetItem.source_item_id.is_(None),
            )
            .order_by(DatasetItem.id)
        )
    ).scalars()
    originals = {i.id: Original(i.id, i.external_id, i.storage_url, i.content_hash) for i in rows}
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


def _has_content(original: Original, modality: str) -> bool:
    if modality == "text":
        return original.text is not None
    if modality == "tabular":
        return isinstance(original.features_json, dict)
    return original.storage_url is not None


def _samples_for_prepare(originals: list[Original], modality: str) -> list[Any]:
    if modality == "text":
        return [o.text["raw_text"] for o in originals if o.text]
    if modality == "tabular":
        return [o.features_json for o in originals if isinstance(o.features_json, dict)]
    return []


# -- Producing copies (blocking; runs in worker threads) -------------------------------------


def _aug_name(external_id: str | None, number: int, ext: str = "") -> str:
    if not external_id:
        return f"aug{number}{ext}"
    stem = external_id
    if ext and "." in external_id:
        stem = external_id.rsplit(".", 1)[0]
    return f"{stem}_aug{number}{ext}"


def _produce(
    original: Original,
    plan: list[ConfiguredOp],
    config: AugmentationConfig,
    version_id: uuid.UUID,
    modality: str,
) -> list[Produced] | None:
    """All copies of one original, or None if it could not be augmented (undecodable file, missing object...).

    Decode, augment and encode failures skip the ORIGINAL; a failed upload is infrastructure trouble and
    propagates so the whole build fails rather than quietly producing a thinner snapshot.
    """
    pending: list[tuple[int, list[dict[str, Any]], bytes | None, str, str, str | None, dict[str, Any]]] = []
    try:
        source_ext = media.file_ext(original.storage_url)
        if modality in ("vision", "audio"):
            raw = storage.download_bytes(C.BUCKET_DATASETS, original.storage_url or "")
            sample = media.decode_image(raw) if modality == "vision" else media.decode_audio(raw, source_ext)
        elif modality == "text":
            sample = original.text["raw_text"] if original.text else ""
        else:
            sample = original.features_json

        for i in range(config.copies_per_item):
            rng = random.Random(seed_for(version_id, original.id, i))
            new, applied = augment(sample, plan, rng)
            number = i + 1
            if modality == "vision":
                enc = media.encode_image(new, source_ext)
            elif modality == "audio":
                enc = media.encode_audio(new)
            else:
                enc = None

            if enc is not None:
                digest = hashlib.sha256(enc.data).hexdigest()
                pending.append((number, applied, enc.data, digest, enc.ext, enc.content_type, enc.features))
            elif modality == "text":
                if new == sample:
                    continue  # the ops could not change this text; a duplicate teaches nothing
                digest = hashlib.sha256(new.encode()).hexdigest()
                features = {
                    "raw_text": new,
                    "token_count": len(new.split()),
                    "language_code": (original.text or {}).get("language_code"),
                    "meta_json": (original.text or {}).get("meta_json"),
                }
                pending.append((number, applied, None, digest, "", None, features))
            else:
                if new == sample:
                    continue
                digest = hashlib.sha256(canonical_json(new).encode()).hexdigest()
                pending.append((number, applied, None, digest, "", None, {"features_json": new}))
    except Exception:
        logger.warning("Could not augment item %s; skipping it", original.id, exc_info=True)
        return None

    produced: list[Produced] = []
    for number, applied, data, digest, ext, content_type, features in pending:
        if data is not None and digest == original.content_hash:
            continue  # re-encoding reproduced the original bytes
        key = None
        if data is not None:
            key = storage.augmented_key(str(version_id), digest, ext)
            storage.upload_bytes(C.BUCKET_DATASETS, key, data, content_type)
        produced.append(
            Produced(
                copy_number=number,
                external_id=_aug_name(original.external_id, number, ext),
                storage_url=key,
                content_hash=digest,
                byte_size=len(data) if data is not None else None,
                features=features,
                applied=applied,
            )
        )
    return produced


# -- Writing ---------------------------------------------------------------------------------


def _feature_row(modality: str, item_id: uuid.UUID, f: dict[str, Any]) -> Any:
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


def _add_items(
    session: AsyncSession, dataset_id: uuid.UUID, original: Original, copies: list[Produced]
) -> list[tuple[uuid.UUID, Produced]]:
    """Stage the item rows for one original's copies; ids are assigned here so children can reference them."""
    staged = []
    for p in copies:
        item_id = uuid.UUID(str(uuid_utils.uuid7()))  # time-ordered, like the DB uuidv7() default
        session.add(
            DatasetItem(
                id=item_id,
                dataset_id=dataset_id,
                external_id=p.external_id,
                storage_url=p.storage_url,
                content_hash=p.content_hash,
                byte_size=p.byte_size,
                source_item_id=original.id,
                augmentation={"copy": p.copy_number, "ops": p.applied},
            )
        )
        staged.append((item_id, p))
    return staged


def _add_children(
    session: AsyncSession,
    version_id: uuid.UUID,
    modality: str,
    original: Original,
    staged: list[tuple[uuid.UUID, Produced]],
) -> None:
    """Features, copied annotations and train-split membership. Must run after the item rows are flushed."""
    for item_id, p in staged:
        session.add(_feature_row(modality, item_id, p.features))
        for a in original.annotations:
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
        session.add(DatasetVersionItem(version_id=version_id, item_id=item_id, split_type="train"))


async def materialize_augmentations(version_id: uuid.UUID) -> AugmentResult:
    """Create the augmented train-split copies for a `building` version. Raises if nothing could be created."""
    sessionmaker = get_sessionmaker()
    loop = asyncio.get_running_loop()

    async with sessionmaker() as session:
        version = await session.get(DatasetVersion, version_id)
        if version is None or not version.augmentation_config:
            return AugmentResult()
        config = AugmentationConfig.model_validate(version.augmentation_config)
        dataset_id = version.dataset_id
        project = await session.get(Project, dataset_id)
        if project is None:
            raise ValueError(f"Version {version_id} project not found")
        modality = get_task_descriptor(project.task).modality
        originals = [o for o in await _load_originals(session, version_id, modality) if _has_content(o, modality)]

    result = AugmentResult(total=len(originals))
    if not originals:
        return result

    plan = build_plan(config, _samples_for_prepare(originals, modality))
    gate = asyncio.Semaphore(_CONCURRENCY)

    async def run(original: Original) -> list[Produced] | None:
        async with gate:
            return await loop.run_in_executor(None, _produce, original, plan, config, version_id, modality)

    for start in range(0, len(originals), _BATCH):
        batch = originals[start : start + _BATCH]
        produced = await asyncio.gather(*(run(o) for o in batch))
        async with sessionmaker() as session:
            staged_by_original: list[tuple[Original, list[tuple[uuid.UUID, Produced]]]] = []
            for original, copies in zip(batch, produced, strict=True):
                if copies is None:
                    result.skipped += 1
                    continue
                staged_by_original.append((original, _add_items(session, dataset_id, original, copies)))
                result.created += len(copies)
            await session.flush()  # items first: the rows below reference them by foreign key
            for original, staged in staged_by_original:
                _add_children(session, version_id, modality, original, staged)
            await session.commit()

    if result.created == 0:
        raise RuntimeError(
            f"The chosen augmentations produced no new items ({result.skipped} of {result.total} originals "
            "could not be processed, the rest were left unchanged by the ops)"
        )
    return result


# -- Removal ---------------------------------------------------------------------------------


async def delete_augmented_items(session: AsyncSession, version_id: uuid.UUID) -> int:
    """Delete a snapshot's augmented copies (rows only; S3 is cleanup_version_storage's job).

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
