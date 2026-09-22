"""Snapshot-time augmentation end to end: real Postgres, stubbed S3, every modality."""

import hashlib
import io
import json
import uuid

import numpy as np
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa
from PIL import Image

from theseus.augmentation import media
from theseus.augmentation.media import AudioClip
from theseus.db.models import (
    Annotation,
    AudioFeatures,
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    TabularFeatures,
    TextFeatures,
    User,
    VisionFeatures,
)
from theseus.services import augmentation as aug_service
from theseus.services import snapshot as snap
from theseus.services import storage

BUCKET = "theseus-datasets"
TEXT = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


@pytest.fixture
def s3(monkeypatch):
    store: dict[str, bytes] = {}
    deleted_prefixes: list[str] = []
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__(k, data))
    monkeypatch.setattr(storage, "download_bytes", lambda b, k: store[k])
    monkeypatch.setattr(storage, "delete_prefix", lambda b, p: deleted_prefixes.append(p) or 0)
    store_api = type("S3", (), {})()
    store_api.objects = store
    store_api.deleted_prefixes = deleted_prefixes
    return store_api


def config(*ops, copies=2):
    return {
        "copiesPerItem": copies,
        "ops": [{"id": i, "probability": 1.0, "params": params} for i, params in ops],
    }


async def seed(db, task, modality, aug_config):
    async with db() as s:
        user = User(name="U", email=f"{uuid.uuid4()}@x.co", password_hash="x")
        s.add(user)
        await s.flush()
        project = Project(user_id=user.id, name="p", task=task)
        s.add(project)
        await s.flush()
        s.add(Dataset(project_id=project.id, modality=modality))
        await s.flush()
        version = DatasetVersion(
            dataset_id=project.id, version_tag="v1", status="building", augmentation_config=aug_config
        )
        s.add(version)
        cats = LabelClass(dataset_id=project.id, name="cat")
        dogs = LabelClass(dataset_id=project.id, name="dog")
        s.add_all([cats, dogs])
        await s.commit()
        return project.id, version.id, cats.class_id, dogs.class_id


async def add_item(db, project_id, version_id, split, *, cls=None, external_id=None, storage_url=None, data=None,
                   text=None, features=None, label=None, image=None, audio=None):  # fmt: skip
    async with db() as s:
        raw = data if data is not None else (text or json.dumps(features or {})).encode()
        item = DatasetItem(
            dataset_id=project_id, external_id=external_id, storage_url=storage_url,
            content_hash=hashlib.sha256(raw + uuid.uuid4().bytes).hexdigest(), byte_size=len(raw),
        )  # fmt: skip
        s.add(item)
        await s.flush()
        s.add(DatasetVersionItem(version_id=version_id, item_id=item.id, split_type=split))
        if text is not None:
            s.add(TextFeatures(item_id=item.id, raw_text=text, token_count=len(text.split()), language_code="en"))
        if features is not None:
            s.add(TabularFeatures(item_id=item.id, features_json=features))
        if image:
            s.add(VisionFeatures(item_id=item.id, width=image[0], height=image[1], channels=3, image_format="png"))
        if audio:
            s.add(
                AudioFeatures(item_id=item.id, duration_seconds=1, sample_rate_hz=8000, channels=1, audio_codec="wav")
            )
        if cls is not None:
            s.add(Annotation(item_id=item.id, annotation_type="classification", class_id=cls))
        if label is not None:
            s.add(Annotation(item_id=item.id, annotation_type="classification", label_structured={"value": label}))
        await s.commit()
        return item.id


def png(color, size=(20, 16)):
    buf = io.BytesIO()
    rng = np.random.default_rng(color)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def wav():
    t = np.linspace(0, 1, 8000, dtype=np.float32)
    return media.encode_audio(AudioClip((0.5 * np.sin(2 * np.pi * 220 * t))[None, :], 8000)).data


async def version_items(db, version_id):
    async with db() as s:
        rows = (
            await s.execute(
                sa.select(DatasetItem, DatasetVersionItem.split_type)
                .join(DatasetVersionItem, DatasetVersionItem.item_id == DatasetItem.id)
                .where(DatasetVersionItem.version_id == version_id)
            )
        ).all()
    return [(i, split) for i, split in rows]


# -- Text ------------------------------------------------------------------------------------


async def test_a_text_snapshot_gets_augmented_train_copies_with_copied_labels_and_untouched_eval_splits(db, s3):
    project_id, version_id, cat, dog = await seed(
        db, "text_classification", "text", config(("text_word_swap", {"swaps": 4}))
    )
    a = await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT, external_id="a.txt")
    b = await add_item(db, project_id, version_id, "train", cls=dog, text=TEXT + " kilo", external_id="b.txt")
    await add_item(db, project_id, version_id, "validation", cls=cat, text=TEXT + " lima")
    await add_item(db, project_id, version_id, "test", cls=dog, text=TEXT + " mike")

    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert (v.augmented_count, v.item_count) == (4, 8)  # 2 train originals x 2 copies, all in the count
    assert v.augmentation_config["copiesPerItem"] == 2

    items = await version_items(db, version_id)
    augmented = [(i, split) for i, split in items if i.source_item_id is not None]
    assert len(augmented) == 4 and {split for _, split in augmented} == {"train"}  # eval splits stay real
    assert {i.source_item_id for i, _ in augmented} == {a, b}
    assert all(i.storage_url is None and i.content_hash and len(i.content_hash) == 64 for i, _ in augmented)
    assert {i.external_id for i, _ in augmented if i.source_item_id == a} == {"a.txt_aug1", "a.txt_aug2"}
    assert all(i.augmentation["ops"] == [{"id": "text_word_swap", "params": {"swaps": 4}}] for i, _ in augmented)
    assert sorted(i.augmentation["copy"] for i, _ in augmented) == [1, 1, 2, 2]

    async with db() as s:
        for item, _ in augmented:
            feats = (await s.execute(sa.select(TextFeatures).where(TextFeatures.item_id == item.id))).scalar_one()
            source_text = {a: TEXT, b: TEXT + " kilo"}[item.source_item_id]
            assert feats.raw_text != source_text  # changed...
            assert sorted(feats.raw_text.split()) == sorted(source_text.split())  # ...by reordering, nothing lost
            assert feats.language_code == "en" and feats.token_count == len(feats.raw_text.split())
            ann = (await s.execute(sa.select(Annotation).where(Annotation.item_id == item.id))).scalar_one()
            want = cat if item.source_item_id == a else dog
            assert (ann.annotation_type, ann.class_id) == ("classification", want)  # the label travels with the copy

    table = pq.read_table(io.BytesIO(s3.objects[f"snapshots/{version_id}/dataset.parquet"]))
    assert table.num_rows == 8
    assert sum(1 for r in table.to_pylist() if r["_ludwig_split_idx"] == 0) == 6  # 2 originals + 4 copies train
    ctx = await snap.read_snapshot_manifest(version_id)
    assert ctx.class_counts == {"cat": 4, "dog": 4}  # per class: 1 train + 2 copies + 1 real eval item


async def test_rebuilding_is_deterministic_for_the_same_snapshot(db, s3):
    project_id, version_id, cat, _ = await seed(
        db, "text_classification", "text", config(("text_word_swap", {}), copies=3)
    )
    await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT)
    await snap.build_snapshot(version_id)
    first = sorted(i.content_hash for i, _ in await version_items(db, version_id) if i.source_item_id)

    # Same snapshot id, same seeds: run the augmentation phase again on a clean slate.
    async with db() as s:
        await aug_service.delete_augmented_items(s, version_id)
        await s.commit()
    result = await aug_service.materialize_augmentations(version_id)
    second = sorted(i.content_hash for i, _ in await version_items(db, version_id) if i.source_item_id)
    assert result.created == len(second) and first == second


async def test_copies_that_the_ops_cannot_change_are_not_created(db, s3):
    project_id, version_id, cat, _ = await seed(
        db, "text_classification", "text", config(("text_word_swap", {}), copies=2)
    )
    await add_item(db, project_id, version_id, "train", cls=cat, text="same same same same")  # swapping changes nothing
    good = await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT)
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.augmented_count == 2
    assert {i.source_item_id for i, _ in await version_items(db, version_id) if i.source_item_id} == {good}


async def test_a_snapshot_where_no_copy_could_be_made_fails_and_leaves_nothing_behind(db, s3):
    project_id, version_id, cat, _ = await seed(
        db, "text_classification", "text", config(("text_word_swap", {}), copies=2)
    )
    await add_item(db, project_id, version_id, "train", cls=cat, text="same same same same")
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "failed" and "produced no new items" in v.failed_message
    assert not [i for i, _ in await version_items(db, version_id) if i.source_item_id]
    assert s3.deleted_prefixes == [f"snapshots/{version_id}/augmented/"]


# -- Vision ----------------------------------------------------------------------------------


async def test_a_vision_snapshot_writes_augmented_images_under_its_own_prefix_with_features(db, s3):
    project_id, version_id, cat, _ = await seed(
        db, "image_classification", "vision", config(("image_horizontal_flip", {}), ("image_gaussian_noise", {}))
    )
    s3.objects["pool/p/aa/a.png"] = png(1)
    original = await add_item(
        db, project_id, version_id, "train", cls=cat, external_id="cat.png", storage_url="pool/p/aa/a.png",
        data=s3.objects["pool/p/aa/a.png"], image=(20, 16),
    )  # fmt: skip
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert v.augmented_count == 2
    copies = [i for i, _ in await version_items(db, version_id) if i.source_item_id == original]
    assert len(copies) == 2
    for item in copies:
        assert item.storage_url.startswith(f"snapshots/{version_id}/augmented/") and item.storage_url.endswith(".png")
        assert item.storage_url.split("/")[-1].startswith(item.content_hash)  # hash-named, in the snapshot prefix
        data = s3.objects[item.storage_url]
        assert hashlib.sha256(data).hexdigest() == item.content_hash and item.byte_size == len(data)
        assert Image.open(io.BytesIO(data)).size == (20, 16)
        async with db() as s:
            f = (await s.execute(sa.select(VisionFeatures).where(VisionFeatures.item_id == item.id))).scalar_one()
        assert (f.width, f.height, f.channels, f.image_format) == (20, 16, 3, "png")
        assert item.external_id.startswith("cat_aug") and item.external_id.endswith(".png")

    rows = pq.read_table(io.BytesIO(s3.objects[f"snapshots/{version_id}/dataset.parquet"])).to_pylist()
    uris = {r["image_path"] for r in rows}
    assert f"s3://{BUCKET}/pool/p/aa/a.png" in uris  # the original keeps pointing into the pool
    assert sum(f"s3://{BUCKET}/snapshots/{version_id}/augmented/" in u for u in uris) == 2


async def test_an_undecodable_original_is_skipped_and_counted_but_does_not_fail_the_snapshot(db, s3):
    project_id, version_id, cat, _ = await seed(db, "image_classification", "vision", config(("image_rotate", {})))
    s3.objects["pool/p/aa/good.png"] = png(2)
    s3.objects["pool/p/bb/bad.png"] = b"this is not an image"
    good = await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/aa/good.png",
                          data=s3.objects["pool/p/aa/good.png"])  # fmt: skip
    await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/bb/bad.png", data=b"bad")
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.augmented_count == 2 and v.augmentation_config["skippedItems"] == 1
    assert {i.source_item_id for i, _ in await version_items(db, version_id) if i.source_item_id} == {good}


async def test_a_failed_build_discards_the_partial_augmentation_rows_and_files(db, s3, monkeypatch):
    project_id, version_id, cat, _ = await seed(db, "image_classification", "vision", config(("image_rotate", {})))
    s3.objects["pool/p/aa/a.png"] = png(3)
    await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/aa/a.png",
                   data=s3.objects["pool/p/aa/a.png"])  # fmt: skip

    real_upload = storage.upload_bytes

    def flaky(bucket, key, data, content_type=None):
        if key.endswith("dataset.parquet"):
            raise RuntimeError("s3 went away")
        return real_upload(bucket, key, data, content_type)

    monkeypatch.setattr(storage, "upload_bytes", flaky)
    await snap.build_snapshot(version_id)  # augmentation succeeds, the parquet upload then fails

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
        remaining = (
            await s.execute(sa.select(sa.func.count()).where(DatasetItem.source_item_id.is_not(None)))
        ).scalar_one()
    assert v.status == "failed" and "s3 went away" in v.failed_message and v.augmented_count == 0
    assert remaining == 0 and s3.deleted_prefixes == [f"snapshots/{version_id}/augmented/"]


# -- Audio -----------------------------------------------------------------------------------


async def test_an_audio_snapshot_writes_wav_copies_with_audio_features(db, s3):
    project_id, version_id, cat, _ = await seed(
        db, "audio_classification", "audio", config(("audio_gain", {"maxDb": 10}), copies=1)
    )
    s3.objects["pool/p/aa/a.wav"] = wav()
    original = await add_item(
        db, project_id, version_id, "train", cls=cat, external_id="bark.wav", storage_url="pool/p/aa/a.wav",
        data=s3.objects["pool/p/aa/a.wav"], audio=True,
    )  # fmt: skip
    await snap.build_snapshot(version_id)

    (item,) = [i for i, _ in await version_items(db, version_id) if i.source_item_id == original]
    assert item.storage_url.endswith(".wav") and item.external_id == "bark_aug1.wav"
    clip = media.decode_audio(s3.objects[item.storage_url], ".wav")
    assert clip.sample_rate == 8000 and clip.samples.shape == (1, 8000)
    async with db() as s:
        f = (await s.execute(sa.select(AudioFeatures).where(AudioFeatures.item_id == item.id))).scalar_one()
    assert (float(f.duration_seconds), f.sample_rate_hz, f.channels, f.audio_codec) == (1.0, 8000, 1, "wav")


# -- Tabular ---------------------------------------------------------------------------------


async def test_a_tabular_regression_snapshot_perturbs_features_and_keeps_the_target(db, s3):
    project_id, version_id, *_ = await seed(
        db, "tabular_regression", "tabular", config(("tabular_gaussian_noise", {"noiseStd": 0.3}), copies=2)
    )
    for age, target in [(30, 1.5), (40, 2.5), (50, 3.5), (60, 4.5)]:
        await add_item(db, project_id, version_id, "train", features={"age": age, "income": age * 1000.5}, label=target)
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
        assert v.status == "ready" and v.augmented_count == 8
        copies = (
            (await s.execute(sa.select(DatasetItem).where(DatasetItem.source_item_id.is_not(None)))).scalars().all()
        )
        for item in copies:
            feats = (await s.execute(sa.select(TabularFeatures).where(TabularFeatures.item_id == item.id))).scalar_one()
            ann = (await s.execute(sa.select(Annotation).where(Annotation.item_id == item.id))).scalar_one()
            source_target = (
                await s.execute(sa.select(Annotation.label_structured).where(Annotation.item_id == item.source_item_id))
            ).scalar_one()
            assert set(feats.features_json) == {"age", "income"} and isinstance(feats.features_json["age"], int)
            assert ann.label_structured == source_target  # the regression target is copied, never perturbed
    rows = pq.read_table(io.BytesIO(s3.objects[f"snapshots/{version_id}/dataset.parquet"])).to_pylist()
    assert len(rows) == 12 and sorted({r["target"] for r in rows}) == [1.5, 2.5, 3.5, 4.5]


# -- No augmentation, and removal ------------------------------------------------------------


async def test_a_snapshot_without_augmentation_is_unchanged(db, s3):
    project_id, version_id, cat, _ = await seed(db, "text_classification", "text", None)
    await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT)
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert (v.status, v.item_count, v.augmented_count, v.augmentation_config) == ("ready", 1, 0, None)
    assert s3.deleted_prefixes == []


async def test_delete_augmented_items_removes_the_copies_but_never_the_originals(db, s3):
    project_id, version_id, cat, _ = await seed(db, "text_classification", "text", config(("text_word_swap", {})))
    original = await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT)
    await snap.build_snapshot(version_id)

    async with db() as s:
        assert await aug_service.delete_augmented_items(s, version_id) == 2
        await s.commit()
    async with db() as s:
        left = (await s.execute(sa.select(DatasetItem.id))).scalars().all()
        assert left == [original]
        # features and annotations of the removed copies cascaded away with them
        assert (await s.execute(sa.select(sa.func.count()).select_from(TextFeatures))).scalar_one() == 1
        assert (await s.execute(sa.select(sa.func.count()).select_from(Annotation))).scalar_one() == 1
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetVersionItem))).scalar_one() == 1


async def test_an_original_with_augmented_copies_cannot_be_hard_deleted(db, s3):
    project_id, version_id, cat, _ = await seed(db, "text_classification", "text", config(("text_word_swap", {})))
    original = await add_item(db, project_id, version_id, "train", cls=cat, text=TEXT)
    await snap.build_snapshot(version_id)
    async with db() as s:
        with pytest.raises(sa.exc.IntegrityError):  # RESTRICT: the service falls back to a soft delete
            await s.execute(sa.delete(DatasetItem).where(DatasetItem.id == original))
