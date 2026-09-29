"""Snapshot preprocessing over HTTP: the options endpoint, creating a preprocessed snapshot, deleting it."""

import asyncio

import pytest
import sqlalchemy as sa

from theseus.db.models import DatasetItem, DatasetVersionItem, TextFeatures
from theseus.services import storage

TEXT = "Hello World"


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


def pre(*ops, splits=None):
    all_splits = ["train", "validation", "test"]
    return {"ops": [{"id": i, "splits": all_splits if splits is None else splits, "params": p} for i, p in ops]}


async def preprocessed_snapshot(c, pid, tag="v1", **kw):
    r = await c.post(
        f"/api/projects/{pid}/versions",
        json={"versionTag": tag, "preprocessing": kw.get("preprocessing") or pre(("text_lowercase", {}))},
    )
    assert r.status_code == 202, r.text
    return r.json()["version"]


async def seeded(c):
    p = await project(c)
    cls = (await c.post(f"/api/projects/{p['id']}/classes", json={"name": "cat"})).json()["class"]["classId"]
    await add_texts(
        c, p["id"], cls,
        (TEXT, "train"), (TEXT + " two", "train"), (TEXT + " val", "validation"), (TEXT + " test", "test"),
    )  # fmt: skip
    return p, cls


# -- The options endpoint --------------------------------------------------------------------


async def test_preprocessing_options_are_the_installed_ops_that_support_the_projects_task(client, new_user):
    c = await new_user()
    p = await project(c, "text_classification")
    body = (await c.get(f"/api/projects/{p['id']}/preprocessing")).json()["preprocessing"]
    assert {a["modality"] for a in body} == {"text"} and "text_lowercase" in [a["id"] for a in body]

    vision = await project(c, "image_classification")
    assert {
        a["modality"] for a in (await c.get(f"/api/projects/{vision['id']}/preprocessing")).json()["preprocessing"]
    } == {"vision"}
    unsupported = await project(c, "token_classification")
    assert (await c.get(f"/api/projects/{unsupported['id']}/preprocessing")).json()["preprocessing"] == []


async def test_preprocessing_options_are_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    assert (await stranger.get(f"/api/projects/{p['id']}/preprocessing")).status_code == 403


# -- Creating a preprocessed snapshot ---------------------------------------------------------


async def test_creating_a_preprocessed_snapshot_replaces_items_and_records_the_config(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    created = await preprocessed_snapshot(c, p["id"], preprocessing=pre(("text_lowercase", {}), splits=["train"]))
    assert created["preprocessingConfig"]["ops"][0]["splits"] == ["train"] and created["preprocessedCount"] == 0

    v = await wait_ready(c, created["id"])
    assert v["status"] == "ready", v["failedMessage"]
    assert v["preprocessedCount"] == 2 and v["itemCount"] == 4  # replaced, never added
    assert {s["splitType"]: s["itemCount"] for s in v["splits"]} == {"train": 2, "validation": 1, "test": 1}

    items = (await c.get(f"/api/projects/{p['id']}/items", params={"versionId": v["id"], "perPage": 50})).json()[
        "items"
    ]
    train_items = [i for i in items if i["splitType"] == "train"]
    other_items = [i for i in items if i["splitType"] != "train"]
    assert all(i["sourceItemId"] is not None for i in train_items)
    assert all(i["preprocessing"]["ops"][0]["id"] == "text_lowercase" for i in train_items)
    assert all(i["sourceItemId"] is None for i in other_items)  # validation/test were not selected

    # A plain snapshot of the same draft has no preprocessing and is unaffected by the preprocessed one.
    plain = await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "plain"})
    pv = await wait_ready(c, plain.json()["version"]["id"])
    assert (pv["preprocessedCount"], pv["preprocessingConfig"], pv["itemCount"]) == (0, None, 4)


async def test_preprocessing_is_rejected_with_a_clear_message_for_unusable_requests(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    url = f"/api/projects/{p['id']}/versions"

    def post(preprocessing, tag="t"):
        return c.post(url, json={"versionTag": tag, "preprocessing": preprocessing})

    msg = lambda r: r.json()["error"]["message"]  # noqa: E731
    assert "Unknown preprocessing op 'nope'" in msg(await post(pre(("nope", {}))))
    assert "not available for Text Classification" in msg(await post(pre(("image_resize", {}))))
    assert (await post({"ops": []})).status_code == 422
    assert (await post(pre(("text_lowercase", {}), splits=[]))).status_code == 422

    empty = await project(c)
    r = await c.post(
        f"/api/projects/{empty['id']}/versions",
        json={"versionTag": "x", "preprocessing": pre(("text_lowercase", {}))},
    )
    assert r.status_code == 400 and "draft is empty" in msg(r)

    # A draft that has items, but none in the op's selected split, hits the preprocessing-specific message.
    no_val = await project(c)
    no_val_cls = (await c.post(f"/api/projects/{no_val['id']}/classes", json={"name": "cat"})).json()["class"][
        "classId"
    ]
    await add_texts(c, no_val["id"], no_val_cls, (TEXT, "train"))
    r = await c.post(
        f"/api/projects/{no_val['id']}/versions",
        json={"versionTag": "x", "preprocessing": pre(("text_lowercase", {}), splits=["validation"])},
    )
    assert r.status_code == 400 and "no items in the selected splits" in msg(r)


async def test_a_rejected_preprocessing_request_creates_no_snapshot(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "bad", "preprocessing": pre(("nope", {}))})
    dataset = (await c.get(f"/api/projects/{p['id']}")).json()["project"]["dataset"]
    assert dataset["versions"] == [] and dataset["draft"]["versionTag"] is None


# -- Composing with augmentation ---------------------------------------------------------------


async def test_preprocessing_and_augmentation_compose_in_one_request(client, new_user):
    c = await new_user()
    p, _ = await seeded(c)
    r = await c.post(
        f"/api/projects/{p['id']}/versions",
        json={
            "versionTag": "v1",
            "preprocessing": pre(("text_lowercase", {}), splits=["train"]),
            "augmentation": {"copiesPerItem": 1, "ops": [{"id": "text_word_swap", "probability": 1.0, "params": {}}]},
        },
    )
    assert r.status_code == 202, r.text
    v = await wait_ready(c, r.json()["version"]["id"])
    assert v["status"] == "ready", v["failedMessage"]
    assert v["preprocessedCount"] == 2 and v["augmentedCount"] == 2 and v["itemCount"] == 6

    items = (await c.get(f"/api/projects/{p['id']}/items", params={"versionId": v["id"], "perPage": 50})).json()[
        "items"
    ]
    augmented = [i for i in items if i["augmentation"] is not None]
    assert all(i["preprocessing"]["ops"][0]["id"] == "text_lowercase" for i in augmented)  # carried forward
    assert all(i["textFeatures"]["rawText"].islower() for i in augmented)


# -- Deleting the snapshot ---------------------------------------------------------------------


async def test_deleting_a_preprocessed_snapshot_removes_its_replacements_and_files_but_keeps_the_pool(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p, _ = await seeded(c)
    created = await preprocessed_snapshot(c, p["id"], preprocessing=pre(("text_lowercase", {})))
    vid = (await wait_ready(c, created["id"]))["id"]
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
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetVersionItem))).scalar_one() == 4
    assert f"snapshots/{vid}/preprocessed/" in fake_s3.prefixes_deleted
    assert (await c.get(f"/api/projects/{p['id']}/items")).json()["total"] == 4
