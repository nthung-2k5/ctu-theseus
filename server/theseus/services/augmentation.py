"""Materialize a snapshot's augmented items: the TRAIN split gets N extra, real, browsable copies.

Runs as the second phase of build_snapshot, after preprocessing (services/preprocessing.py) has
already replaced any train items its ops were scoped to: an augmented copy is made from whatever
is in the train split at that point, so it carries forward a preprocessed item's `preprocessing`
record and always points `source_item_id` at the pool ancestor, never at an intermediate
preprocessed row (see services/derived_items.py).

Each copy is a dataset_items row (with `source_item_id` pointing at its pool ancestor, plus the
modality features and a copy of the original's annotations) and a dataset_version_items row in the
train split, so the parquet, manifest, split counts and the items browser all pick it up with no
special casing. File-backed copies are written under the snapshot's own S3 prefix, never the
content-addressed pool (see storage.augmented_prefix).

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
from dataclasses import dataclass
from typing import Any

import uuid_utils
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.augmentation import media
from theseus.augmentation.config import AugmentationConfig
from theseus.augmentation.pipeline import ConfiguredOp, augment, build_plan, seed_for
from theseus.db.base import get_sessionmaker
from theseus.db.models import DatasetItem, DatasetVersion, DatasetVersionItem, Project
from theseus.services import storage
from theseus.services.datasets import canonical_json
from theseus.services.derived_items import (
    Original,
    copy_annotations,
    feature_row,
    has_content,
    load_items,
    samples_for_fit,
)
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)

_BATCH = 16  # originals per DB transaction
_CONCURRENCY = 4  # originals augmented at once (threads)


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
                # Always the pool ancestor, even when `original` is itself a preprocessed item:
                # there is never a chain of derived items (see db/models/dataset.py).
                source_item_id=original.source_item_id or original.id,
                preprocessing=original.preprocessing,
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
        session.add(feature_row(modality, item_id, p.features))
        copy_annotations(session, item_id, original.annotations)
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
        # Whatever is currently in the train split and not already an augmented copy: a pool
        # original, or a preprocessed replacement if preprocessing ran first.
        originals = [
            o
            for o in await load_items(
                session, version_id, modality, splits={"train"}, where=DatasetItem.augmentation.is_(None)
            )
            if has_content(o, modality)
        ]

    result = AugmentResult(total=len(originals))
    if not originals:
        return result

    plan = build_plan(config, samples_for_fit(originals, modality))
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
