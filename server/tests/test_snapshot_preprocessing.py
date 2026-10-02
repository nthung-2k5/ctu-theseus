"""Snapshot-time preprocessing end to end: real Postgres, stubbed S3, every modality, and its
interaction with augmentation (preprocess first, then augment the preprocessed train items)."""

import hashlib
import io
import json
import uuid

import numpy as np
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa
from PIL import Image

from theseus.db.models import (
    Annotation,
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
from theseus.services import snapshot as snap
from theseus.services import storage
from theseus.services.derived_items import delete_derived_items

BUCKET = "theseus-datasets"


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


def pre_config(*ops):
    return {
        "ops": [
            {"id": i, "splits": kw.get("splits", ["train", "validation", "test"]), "params": kw.get("params", {})}
            for i, kw in ops
        ]
    }


async def seed(db, task, modality, pre, aug=None):
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
            dataset_id=project.id, version_tag="v1", status="building",
            preprocessing_config=pre, augmentation_config=aug,
        )  # fmt: skip
        s.add(version)
        cat = LabelClass(dataset_id=project.id, name="cat")
        s.add(cat)
        await s.commit()
        return project.id, version.id, cat.class_id


async def add_item(db, project_id, version_id, split, *, cls=None, external_id=None, storage_url=None, data=None,
                   text=None, features=None, image=None):  # fmt: skip
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
        if cls is not None:
            s.add(Annotation(item_id=item.id, annotation_type="classification", class_id=cls))
        await s.commit()
        return item.id


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


def png(color, size=(20, 16)):
    buf = io.BytesIO()
    rng = np.random.default_rng(color)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# -- Text ------------------------------------------------------------------------------------


async def test_lowercasing_train_only_replaces_train_items_and_leaves_other_splits_real(db, s3):
    project_id, version_id, cat = await seed(
        db, "text_classification", "text", pre_config(("text_lowercase", {"splits": ["train"]}))
    )
    a = await add_item(db, project_id, version_id, "train", cls=cat, text="Hello World", external_id="a.txt")
    b = await add_item(db, project_id, version_id, "validation", cls=cat, text="Hello Eval")

    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert (v.preprocessed_count, v.item_count) == (1, 2)  # replace, never add
    assert v.preprocessing_config["ops"] == [{"id": "text_lowercase", "splits": ["train"], "params": {}}]

    items = await version_items(db, version_id)
    train_item, train_split = next((i, s) for i, s in items if s == "train")
    val_item, val_split = next((i, s) for i, s in items if s == "validation")
    assert train_item.id != a and train_item.source_item_id == a and train_item.augmentation is None
    assert train_item.preprocessing == {"ops": [{"id": "text_lowercase", "params": {}}]}
    assert val_item.id == b and val_item.source_item_id is None  # validation was not selected: untouched

    async with db() as s:
        feats = (await s.execute(sa.select(TextFeatures).where(TextFeatures.item_id == train_item.id))).scalar_one()
        assert feats.raw_text == "hello world"
        ann = (await s.execute(sa.select(Annotation).where(Annotation.item_id == train_item.id))).scalar_one()
        assert (ann.annotation_type, ann.class_id) == ("classification", cat)  # the label travelled with the copy


async def test_items_the_ops_cannot_change_are_left_as_the_pool_item(db, s3):
    project_id, version_id, cat = await seed(db, "text_classification", "text", pre_config(("text_lowercase", {})))
    a = await add_item(db, project_id, version_id, "train", cls=cat, text="already lowercase")
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.preprocessed_count == 0
    items = await version_items(db, version_id)
    assert [i.id for i, _ in items] == [a]  # nothing was swapped


# -- Vision ----------------------------------------------------------------------------------


async def test_a_vision_snapshot_writes_resized_images_under_its_own_prefix_with_features(db, s3):
    project_id, version_id, cat = await seed(
        db, "image_classification", "vision", pre_config(("image_resize", {"params": {"width": 8, "height": 8}}))
    )
    s3.objects["pool/p/aa/a.png"] = png(1, size=(20, 16))
    original = await add_item(
        db, project_id, version_id, "train", cls=cat, external_id="cat.png", storage_url="pool/p/aa/a.png",
        data=s3.objects["pool/p/aa/a.png"], image=(20, 16),
    )  # fmt: skip
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert v.preprocessed_count == 1
    (item,) = [i for i, _ in await version_items(db, version_id) if i.source_item_id == original]
    assert item.storage_url.startswith(f"snapshots/{version_id}/preprocessed/") and item.storage_url.endswith(".png")
    assert item.external_id == "cat.png"  # a replacement keeps the original's name
    data = s3.objects[item.storage_url]
    assert Image.open(io.BytesIO(data)).size == (8, 8)
    async with db() as s:
        f = (await s.execute(sa.select(VisionFeatures).where(VisionFeatures.item_id == item.id))).scalar_one()
    assert (f.width, f.height) == (8, 8)

    rows = pq.read_table(io.BytesIO(s3.objects[f"snapshots/{version_id}/dataset.parquet"])).to_pylist()
    uris = {r["image_path"] for r in rows}
    assert f"s3://{BUCKET}/{item.storage_url}" in uris
    assert f"s3://{BUCKET}/pool/p/aa/a.png" not in uris  # the original is no longer a member


async def test_an_undecodable_original_is_skipped_and_counted_but_does_not_fail_the_snapshot(db, s3):
    project_id, version_id, cat = await seed(db, "image_classification", "vision", pre_config(("image_resize", {})))
    s3.objects["pool/p/aa/good.png"] = png(2)
    s3.objects["pool/p/bb/bad.png"] = b"not an image"
    good = await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/aa/good.png",
                          data=s3.objects["pool/p/aa/good.png"])  # fmt: skip
    await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/bb/bad.png", data=b"bad")
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.preprocessed_count == 1 and v.preprocessing_config["skippedItems"] == 1
    assert {i.source_item_id for i, _ in await version_items(db, version_id) if i.source_item_id} == {good}


async def test_a_failed_build_discards_the_partial_preprocessing_rows_and_files(db, s3, monkeypatch):
    project_id, version_id, cat = await seed(db, "image_classification", "vision", pre_config(("image_resize", {})))
    s3.objects["pool/p/aa/a.png"] = png(3)
    await add_item(db, project_id, version_id, "train", cls=cat, storage_url="pool/p/aa/a.png",
                   data=s3.objects["pool/p/aa/a.png"])  # fmt: skip

    real_upload = storage.upload_bytes

    def flaky(bucket, key, data, content_type=None):
        if key.endswith("dataset.parquet"):
            raise RuntimeError("s3 went away")
        return real_upload(bucket, key, data, content_type)

    monkeypatch.setattr(storage, "upload_bytes", flaky)
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
        remaining = (
            await s.execute(sa.select(sa.func.count()).where(DatasetItem.source_item_id.is_not(None)))
        ).scalar_one()
    assert v.status == "failed" and "s3 went away" in v.failed_message and v.preprocessed_count == 0
    assert remaining == 0 and s3.deleted_prefixes == [f"snapshots/{version_id}/preprocessed/"]


# -- Tabular: fit on train even when train is not itself preprocessed ------------------------


async def test_tabular_standardize_fits_on_train_even_when_only_test_is_selected(db, s3):
    project_id, version_id, cat = await seed(
        db, "tabular_classification", "tabular", pre_config(("tabular_standardize", {"splits": ["test"]}))
    )
    train_ages = [0, 10, 20, 30, 40]
    for age in train_ages:
        await add_item(db, project_id, version_id, "train", cls=cat, features={"age": age})
    test_id = await add_item(db, project_id, version_id, "test", cls=cat, features={"age": 30})

    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready" and v.preprocessed_count == 1  # only the test item was replaced

    items = await version_items(db, version_id)
    assert all(i.source_item_id is None for i, split in items if split == "train")  # train stays real
    (replacement,) = [i for i, split in items if split == "test"]
    assert replacement.source_item_id == test_id
    async with db() as s:
        feats = (
            await s.execute(sa.select(TabularFeatures).where(TabularFeatures.item_id == replacement.id))
        ).scalar_one()
    assert feats.features_json["age"] == pytest.approx((30 - np.mean(train_ages)) / np.std(train_ages), rel=1e-6)


# -- Interaction with augmentation ------------------------------------------------------------


async def test_augmentation_is_built_from_the_preprocessed_train_item_not_the_raw_original(db, s3):
    project_id, version_id, cat = await seed(
        db,
        "text_classification",
        "text",
        pre_config(("text_lowercase", {"splits": ["train"]})),
        aug={"copiesPerItem": 1, "ops": [{"id": "text_word_swap", "probability": 1.0, "params": {"swaps": 3}}]},
    )
    original = await add_item(
        db, project_id, version_id, "train", cls=cat, text="Hello World Foo Bar Baz Qux", external_id="a.txt"
    )
    await snap.build_snapshot(version_id)

    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert v.status == "ready", v.failed_message
    assert (v.preprocessed_count, v.augmented_count, v.item_count) == (1, 1, 2)

    items = await version_items(db, version_id)
    preprocessed = next(i for i, _ in items if i.source_item_id == original and i.augmentation is None)
    augmented = next(i for i, _ in items if i.augmentation is not None)

    # source_item_id always points at the pool original directly, never at the preprocessed row.
    assert augmented.source_item_id == original
    # ...but the augmented copy carries forward the preprocessing record it inherited.
    assert augmented.preprocessing == preprocessed.preprocessing == {"ops": [{"id": "text_lowercase", "params": {}}]}

    async with db() as s:
        feats = (await s.execute(sa.select(TextFeatures).where(TextFeatures.item_id == augmented.id))).scalar_one()
    # the augmented copy's words are a subset of the LOWERCASED text, never the original casing
    assert feats.raw_text.islower()
    assert set(feats.raw_text.split()) == {"hello", "world", "foo", "bar", "baz", "qux"}


# -- No preprocessing, and removal ------------------------------------------------------------


async def test_a_snapshot_without_preprocessing_is_unchanged(db, s3):
    project_id, version_id, cat = await seed(db, "text_classification", "text", None)
    await add_item(db, project_id, version_id, "train", cls=cat, text="Hello World")
    await snap.build_snapshot(version_id)
    async with db() as s:
        v = await s.get(DatasetVersion, version_id)
    assert (v.status, v.item_count, v.preprocessed_count, v.preprocessing_config) == ("ready", 1, 0, None)
    assert s3.deleted_prefixes == []


async def test_deleting_a_preprocessed_snapshots_derived_items_keeps_the_pool_original(db, s3):
    project_id, version_id, cat = await seed(db, "text_classification", "text", pre_config(("text_lowercase", {})))
    original = await add_item(db, project_id, version_id, "train", cls=cat, text="Hello World")
    await snap.build_snapshot(version_id)

    async with db() as s:
        # Unlike augmentation, preprocessing REPLACES the item's single membership row rather than
        # adding one: deleting the derived item leaves this version with none, same as the real
        # delete-version route (which deletes the version row itself right after, cascading the rest).
        assert await delete_derived_items(s, version_id) == 1
        await s.commit()
    async with db() as s:
        left = (await s.execute(sa.select(DatasetItem.id))).scalars().all()
        assert left == [original]  # the pool original is untouched
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetVersionItem))).scalar_one() == 0
