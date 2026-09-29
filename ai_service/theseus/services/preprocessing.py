"""Materialize a snapshot's preprocessing: replace pool items with a deterministic, preprocessed
copy for whichever splits (train / validation / test) each op was configured on.

Runs as the first phase of build_snapshot, before augmentation. Unlike augmentation, nothing is
ADDED: a preprocessed item is a real dataset_items row (`source_item_id` pointing at the pool
original) that REPLACES the original in the version's membership for exactly the splits its ops
selected — an item whose split none of the ops selected, or whose output turns out byte-identical
to the original, is simply left as the pool item it already is. `fit`-needing ops (tabular
standardize/min-max) always fit on the train split's samples, even if train itself was not among
the splits an op runs on, so nothing about validation or test ever leaks into how a value is fit.

The work per item (download, decode, preprocess, encode, upload) is blocking and CPU-heavy, so it
runs in worker threads with bounded concurrency; database writes happen on the event loop, one
transaction per batch.
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
import uuid_utils

from theseus import constants as C
from theseus.augmentation import media
from theseus.db.base import get_sessionmaker
from theseus.db.models import DatasetItem, DatasetVersion, DatasetVersionItem, Project
from theseus.preprocessing.config import PreprocessingConfig
from theseus.preprocessing.pipeline import ConfiguredOp, build_plan, ops_for_split, preprocess
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

_BATCH = 16  # items per DB transaction
_CONCURRENCY = 4  # items preprocessed at once (threads)


@dataclass
class Produced:
    storage_url: str | None
    content_hash: str
    byte_size: int | None
    # Modality feature-row columns for the new item.
    features: dict[str, Any]
    applied: list[dict[str, Any]]
    # True when no op applied, or the ops left the item byte-identical: the original stays in place.
    unchanged: bool = False


@dataclass
class PreprocessResult:
    created: int = 0  # items whose membership was replaced by a preprocessed copy
    skipped: int = 0  # items that could not be preprocessed (e.g. undecodable file)
    total: int = 0  # items attempted (i.e. in at least one op's selected splits)


def _produce(original: Original, ops: list[ConfiguredOp], version_id: uuid.UUID, modality: str) -> Produced | None:
    """One item's preprocessed replacement, `unchanged` if its ops left it identical, or None if it
    could not be decoded (missing/corrupt file: the original is left in place and counted as skipped).

    A failed upload is infrastructure trouble and propagates so the whole build fails rather than
    quietly producing a thinner snapshot.
    """
    if not ops:
        return Produced(None, "", None, {}, [], unchanged=True)
    try:
        source_ext = media.file_ext(original.storage_url)
        if modality in ("vision", "audio"):
            raw = storage.download_bytes(C.BUCKET_DATASETS, original.storage_url or "")
            sample = media.decode_image(raw) if modality == "vision" else media.decode_audio(raw, source_ext)
        elif modality == "text":
            sample = original.text["raw_text"] if original.text else ""
        else:
            sample = original.features_json

        new, applied = preprocess(sample, ops)

        if modality == "vision":
            enc = media.encode_image(new, source_ext)
        elif modality == "audio":
            enc = media.encode_audio(new)
        else:
            enc = None

        if enc is not None:
            digest = hashlib.sha256(enc.data).hexdigest()
            if digest == original.content_hash:
                return Produced(None, "", None, {}, [], unchanged=True)  # re-encoding reproduced the original bytes
            key = storage.preprocessed_key(str(version_id), digest, enc.ext)
            storage.upload_bytes(C.BUCKET_DATASETS, key, enc.data, enc.content_type)
            return Produced(key, digest, len(enc.data), enc.features, applied)

        if modality == "text":
            if new == sample:
                return Produced(None, "", None, {}, [], unchanged=True)
            digest = hashlib.sha256(new.encode()).hexdigest()
            features = {
                "raw_text": new,
                "token_count": len(new.split()),
                "language_code": (original.text or {}).get("language_code"),
                "meta_json": (original.text or {}).get("meta_json"),
            }
            return Produced(None, digest, None, features, applied)

        if new == sample:
            return Produced(None, "", None, {}, [], unchanged=True)
        digest = hashlib.sha256(canonical_json(new).encode()).hexdigest()
        return Produced(None, digest, None, {"features_json": new}, applied)
    except Exception:
        logger.warning("Could not preprocess item %s; leaving it as the original", original.id, exc_info=True)
        return None


async def materialize_preprocessing(version_id: uuid.UUID) -> PreprocessResult:
    """Replace pool items with their preprocessed copy, per selected split, for a `building` version."""
    sessionmaker = get_sessionmaker()
    loop = asyncio.get_running_loop()

    async with sessionmaker() as session:
        version = await session.get(DatasetVersion, version_id)
        if version is None or not version.preprocessing_config:
            return PreprocessResult()
        config = PreprocessingConfig.model_validate(version.preprocessing_config)
        dataset_id = version.dataset_id
        project = await session.get(Project, dataset_id)
        if project is None:
            raise ValueError(f"Version {version_id} project not found")
        modality = get_task_descriptor(project.task).modality

        selected_splits = {s for op in config.ops for s in op.splits}
        # The train split is always loaded too (even if no op selects it): a fit-needing op
        # (tabular standardize/min-max) fits on train regardless of which splits it runs on.
        loaded = [
            o
            for o in await load_items(
                session,
                version_id,
                modality,
                splits=selected_splits | {"train"},
                where=DatasetItem.source_item_id.is_(None),
            )  # fmt: skip
            if has_content(o, modality)
        ]
        train_items = [o for o in loaded if o.split_type == "train"]
        items = [o for o in loaded if o.split_type in selected_splits]

    result = PreprocessResult(total=len(items))
    if not items:
        return result

    plan = build_plan(config, samples_for_fit(train_items, modality))
    gate = asyncio.Semaphore(_CONCURRENCY)

    async def run(item: Original) -> Produced | None:
        ops = ops_for_split(plan, item.split_type)
        async with gate:
            return await loop.run_in_executor(None, _produce, item, ops, version_id, modality)

    for start in range(0, len(items), _BATCH):
        batch = items[start : start + _BATCH]
        produced = await asyncio.gather(*(run(o) for o in batch))
        async with sessionmaker() as session:
            staged: list[tuple[Original, uuid.UUID, Produced]] = []
            for original, outcome in zip(batch, produced, strict=True):
                if outcome is None:
                    result.skipped += 1
                    continue
                if outcome.unchanged:
                    continue  # left as the pool item; nothing to write, nothing to swap
                new_id = uuid.UUID(str(uuid_utils.uuid7()))  # time-ordered, like the DB uuidv7() default
                session.add(
                    DatasetItem(
                        id=new_id,
                        dataset_id=dataset_id,
                        external_id=original.external_id,
                        storage_url=outcome.storage_url,
                        content_hash=outcome.content_hash,
                        byte_size=outcome.byte_size,
                        source_item_id=original.id,
                        preprocessing={"ops": outcome.applied},
                    )
                )
                staged.append((original, new_id, outcome))
                result.created += 1
            await session.flush()  # items first: the rows below reference them by foreign key
            for original, new_id, outcome in staged:
                session.add(feature_row(modality, new_id, outcome.features))
                copy_annotations(session, new_id, original.annotations)
                # Replace, don't add: the preprocessed item takes over the original's membership row.
                await session.execute(
                    sa.update(DatasetVersionItem)
                    .where(DatasetVersionItem.version_id == version_id, DatasetVersionItem.item_id == original.id)
                    .values(item_id=new_id)
                )
            await session.commit()
    return result
