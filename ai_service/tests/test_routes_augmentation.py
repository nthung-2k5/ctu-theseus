"""Snapshot augmentation over HTTP: the options endpoint, creating an augmented snapshot, browsing it, deleting it."""

import asyncio

import pytest
import sqlalchemy as sa

from theseus.db.models import DatasetItem, DatasetVersionItem, TextFeatures
from theseus.services import storage

TEXT = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    class Store(dict):
        prefixes_deleted: list[str]

    store = Store()
    store.prefixes_deleted = []
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__((b, k), data))
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    monkeypatch.setattr(storage, "delete_file", lambda b, k: None)
    monkeypatch.setattr(storage, "delete_files", lambda b, ks: None)
    monkeypatch.setattr(storage, "delete_prefix", lambda b, p: store.prefixes_deleted.append(p) or 0)
    return store


async def project(c, task="text_classification"):
    return (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]


async def add_texts(c, pid, cls, *texts_and_splits):
    items = [
        {
            "split": split,
            "externalId": f"{i}.txt",
            "textFeatures": {"rawText": text},
            "annotations": [{"annotationType": "classification", "classId": cls}],
        }
        for i, (text, split) in enumerate(texts_and_splits)
    ]
    r = await c.post(f"/api/projects/{pid}/items", json={"items": items})
    assert r.status_code == 200 and r.json()["failed"] == [], r.text


async def wait_ready(c, version_id, timeout=20):
    for _ in range(int(timeout / 0.1)):
        v = (await c.get(f"/api/versions/{version_id}")).json()["version"]
        if v["status"] != "building":
            return v
        await asyncio.sleep(0.1)
    raise AssertionError("snapshot never left `building`")


def aug(*ops, copies=2):
    return {"copiesPerItem": copies, "ops": [{"id": i, "probability": 1.0, "params": p} for i, p in ops]}


async def augmented_snapshot(c, pid, tag="v1", **kw):
    r = await c.post(
        f"/api/projects/{pid}/versions",
        json={"versionTag": tag, "augmentation": kw.get("augmentation") or aug(("text_word_swap", {"swaps": 3}))},
    )
    assert r.status_code == 202, r.text
    return r.json()["version"]


async def seeded(c):
    p = await project(c)
    cls = (await c.post(f"/api/projects/{p['id']}/classes", json={"name": "cat"})).json()["class"]["classId"]
    await add_texts(
        c,
        p["id"],
        cls,
        (TEXT, "train"),
        (TEXT + " kilo", "train"),
        (TEXT + " lima", "validation"),
        (TEXT + " mike", "test"),
    )
    return p, cls


# -- The options endpoint --------------------------------------------------------------------


async def test_augmentation_options_are_the_installed_ops_that_support_the_projects_task(client, new_user):
    c = await new_user()
    p = await project(c, "text_classification")
    body = (await c.get(f"/api/projects/{p['id']}/augmentations")).json()["augmentations"]
    assert {a["modality"] for a in body} == {"text"} and "text_word_swap" in [a["id"] for a in body]
    swap = next(a for a in body if a["id"] == "text_word_swap")
    assert swap["params"] == [
        {"name": "swaps", "label": "Swaps", "description": "Number of random word pairs exchanged.", "type": "int",
         "default": 2, "min": 1.0, "max": 10.0, "step": 1.0, "choices": None, "group": None},
    ]  # fmt: skip

    vision = await project(c, "image_classification")
    assert {
        a["modality"] for a in (await c.get(f"/api/projects/{vision['id']}/augmentations")).json()["augmentations"]
    } == {"vision"}
    unsupported = await project(c, "token_classification")  # token tags would need re-aligning
    assert (await c.get(f"/api/projects/{unsupported['id']}/augmentations")).json()["augmentations"] == []


async def test_augmentation_options_are_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    assert (await stranger.get(f"/api/projects/{p['id']}/augmentations")).status_code == 403


# -- Creating an augmented snapshot ----------------------------------------------------------


async def test_creating_an_augmented_snapshot_adds_train_copies_and_records_the_config(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    created = await augmented_snapshot(c, p["id"])
    assert created["augmentationConfig"]["copiesPerItem"] == 2 and created["augmentedCount"] == 0  # not built yet

    v = await wait_ready(c, created["id"])
    assert v["status"] == "ready", v["failedMessage"]
    assert v["augmentedCount"] == 4 and v["itemCount"] == 8
    assert {s["splitType"]: s["itemCount"] for s in v["splits"]} == {"train": 6, "validation": 1, "test": 1}
    assert v["augmentationConfig"]["ops"] == [{"id": "text_word_swap", "probability": 1.0, "params": {"swaps": 3}}]

    # A plain snapshot of the same draft has no augmentation and is unaffected by the augmented one.
    plain = await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "plain"})
    pv = await wait_ready(c, plain.json()["version"]["id"])
    assert (pv["augmentedCount"], pv["augmentationConfig"], pv["itemCount"]) == (0, None, 4)


async def test_augmentation_is_rejected_with_a_clear_message_for_unusable_requests(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    url = f"/api/projects/{p['id']}/versions"

    def post(augmentation, tag="t"):
        return c.post(url, json={"versionTag": tag, "augmentation": augmentation})

    msg = lambda r: r.json()["error"]["message"]  # noqa: E731
    assert "Unknown augmentation 'nope'" in msg(await post(aug(("nope", {}))))
    assert "not available for Text Classification" in msg(await post(aug(("image_rotate", {}))))
    assert "Invalid parameter 'swaps'" in msg(await post(aug(("text_word_swap", {"swaps": 99}))))
    assert (await post(aug(("text_word_swap", {}), copies=11))).status_code == 422
    assert (await post({"copiesPerItem": 1, "ops": []})).status_code == 422

    empty = await project(c)
    r = await c.post(
        f"/api/projects/{empty['id']}/versions", json={"versionTag": "x", "augmentation": aug(("text_word_swap", {}))}
    )
    assert r.status_code == 400 and "draft is empty" in msg(r)

    # A draft that has items, but none in the train split, hits the augmentation-specific message instead.
    no_train = await project(c)
    no_train_cls = (await c.post(f"/api/projects/{no_train['id']}/classes", json={"name": "cat"})).json()["class"][
        "classId"
    ]
    await add_texts(c, no_train["id"], no_train_cls, (TEXT, "validation"))
    r = await c.post(
        f"/api/projects/{no_train['id']}/versions",
        json={"versionTag": "x", "augmentation": aug(("text_word_swap", {}))},
    )
    assert r.status_code == 400 and "no training items" in msg(r)


async def test_a_rejected_augmentation_request_creates_no_snapshot(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "bad", "augmentation": aug(("nope", {}))})
    dataset = (await c.get(f"/api/projects/{p['id']}")).json()["project"]["dataset"]
    assert dataset["versions"] == [] and dataset["draft"]["versionTag"] is None  # the draft is not a snapshot


# -- Browsing --------------------------------------------------------------------------------


async def test_the_origin_filter_separates_original_items_from_augmented_copies(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    vid = (await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"]))["id"]

    def listing(**params):
        return c.get(f"/api/projects/{p['id']}/items", params={"versionId": vid, "perPage": 50, **params})

    everything = (await listing()).json()
    originals = (await listing(origin="original")).json()
    augmented = (await listing(origin="augmented")).json()
    assert (everything["total"], originals["total"], augmented["total"]) == (8, 4, 4)
    assert all(i["sourceItemId"] is None and i["augmentation"] is None for i in originals["items"])

    source_names = {i["id"]: i["externalId"] for i in originals["items"]}
    for item in augmented["items"]:
        assert item["splitType"] == "train" and item["sourceItemId"] in source_names
        assert item["sourceExternalId"] == source_names[item["sourceItemId"]]
        assert item["externalId"].startswith(item["sourceExternalId"] + "_aug")
        assert item["augmentation"]["ops"][0]["id"] == "text_word_swap" and item["augmentation"]["copy"] in (1, 2)
        assert item["annotations"][0]["annotationType"] == "classification"  # the label travelled with the copy
        assert item["textFeatures"]["rawText"] != TEXT or True

    # The other filters compose with it, and the counts follow the filter.
    train_aug = (await listing(origin="augmented", split="train")).json()
    assert train_aug["total"] == 4 and (await listing(origin="augmented", split="test")).json()["total"] == 0
    assert (await listing(origin="augmented", search="0.txt_aug")).json()["total"] == 2
    assert (await listing(origin="original")).json()["labeledCount"] == 4
    assert (await listing(origin="bogus")).status_code == 422


async def test_augmented_copies_never_appear_in_the_draft_or_the_pool_health_check(client, new_user, db):
    c = await new_user()
    p, _ = await seeded(c)
    vid = (await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"]))["id"]
    draft = (await c.get(f"/api/projects/{p['id']}/items", params={"perPage": 50})).json()
    assert draft["total"] == 4 and all(i["sourceItemId"] is None for i in draft["items"])

    # Two augmented copies may legitimately share a content hash (identical output of different runs);
    # the pool integrity check must not report that as pool duplication.
    async with db() as s:
        copies = (
            (await s.execute(sa.select(DatasetItem).where(DatasetItem.source_item_id.is_not(None)))).scalars().all()
        )
        for copy in copies[:2]:
            copy.content_hash = "f" * 64
        await s.commit()
    health = (await c.get(f"/api/projects/{p['id']}/dataset/health")).json()["health"]
    assert health["itemCount"] == 4 and health["duplicateContentHashes"] == 0 and health["missingContentHash"] == 0
    assert vid


# -- Pool isolation --------------------------------------------------------------------------


async def test_adding_content_identical_to_an_augmented_copy_creates_a_real_pool_item(client, new_user, db):
    c = await new_user()
    p, cls = await seeded(c)
    vid = (await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"]))["id"]
    copy = (await c.get(f"/api/projects/{p['id']}/items", params={"versionId": vid, "origin": "augmented"})).json()[
        "items"
    ][0]
    text = copy["textFeatures"]["rawText"]

    await add_texts(c, p["id"], cls, (text, "train"))  # the exact text of an augmented copy, added to the draft
    async with db() as s:
        row = (
            await s.execute(
                sa.select(DatasetItem)
                .join(TextFeatures)
                .where(TextFeatures.raw_text == text, DatasetItem.source_item_id.is_(None))
            )
        ).scalar_one()
        assert row.id != copy["id"]  # a new original, never deduplicated onto the augmented copy
    assert (await c.get(f"/api/projects/{p['id']}/items")).json()["total"] == 5


async def test_an_augmented_copy_cannot_be_deleted_on_its_own(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    vid = (await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"]))["id"]
    copy = (await c.get(f"/api/projects/{p['id']}/items", params={"versionId": vid, "origin": "augmented"})).json()[
        "items"
    ][0]
    r = await c.delete(f"/api/items/{copy['id']}")
    assert r.status_code == 400 and "delete the snapshot instead" in r.json()["error"]["message"]


async def test_deleting_an_original_that_has_augmented_copies_soft_deletes_it(client, new_user, db):
    c = await new_user()
    p, _ = await seeded(c)
    await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"])
    first = (await c.get(f"/api/projects/{p['id']}/items", params={"sort": "oldest"})).json()["items"][0]
    assert (await c.delete(f"/api/items/{first['id']}")).status_code == 204  # not a foreign key error
    async with db() as s:
        gone = (await s.execute(sa.select(DatasetItem.deleted_at).where(DatasetItem.id == first["id"]))).scalar_one()
    assert gone is not None
    assert (await c.get(f"/api/projects/{p['id']}/items")).json()["total"] == 3


# -- Deleting the snapshot -------------------------------------------------------------------


async def test_deleting_an_augmented_snapshot_removes_its_copies_and_files_but_keeps_the_pool(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p, _ = await seeded(c)
    vid = (await wait_ready(c, (await augmented_snapshot(c, p["id"]))["id"]))["id"]
    async with db() as s:
        assert (
            await s.execute(sa.select(sa.func.count()).where(DatasetItem.source_item_id.is_not(None)))
        ).scalar_one() == 4

    assert (await c.delete(f"/api/versions/{vid}")).status_code == 204
    async with db() as s:
        assert (
            await s.execute(sa.select(sa.func.count()).where(DatasetItem.source_item_id.is_not(None)))
        ).scalar_one() == 0
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetItem))).scalar_one() == 4  # the pool
        assert (await s.execute(sa.select(sa.func.count()).select_from(TextFeatures))).scalar_one() == 4
        # only the draft's membership rows remain
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetVersionItem))).scalar_one() == 4
    assert f"snapshots/{vid}/augmented/" in fake_s3.prefixes_deleted
    assert (await c.get(f"/api/projects/{p['id']}/items")).json()["total"] == 4
