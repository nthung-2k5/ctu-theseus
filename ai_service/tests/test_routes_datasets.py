"""Dataset pool over HTTP: items, dedup, listing, upload, snapshot versions, deletion.

Ports server/routes/datasets.integration.test.ts and covers the file-upload and S3 paths it could not.
"""

import asyncio
import struct
import uuid

import pytest
import sqlalchemy as sa

from theseus.db.models import Annotation, DatasetItem, DatasetVersion, TextFeatures, VisionFeatures
from theseus.routers import datasets as datasets_router
from theseus.services import storage


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    class Store(dict):
        deleted: list[tuple[str, str]]
        puts: list[tuple[str, str]]

    store = Store()
    store.puts = []
    deleted: list[tuple[str, str]] = []

    def put(b, k, data, content_type=None):
        store.puts.append((b, k))
        store[(b, k)] = data

    monkeypatch.setattr(storage, "upload_bytes", put)
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    monkeypatch.setattr(storage, "delete_file", lambda b, k: deleted.append((b, k)))
    monkeypatch.setattr(storage, "delete_files", lambda b, ks: deleted.extend((b, k) for k in ks))
    monkeypatch.setattr(storage, "delete_prefix", lambda b, p: 0)
    store.deleted = deleted
    return store


def png(width: int, height: int) -> bytes:
    """Just enough of a PNG for the header reader: signature, IHDR length, IHDR, width, height."""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + bytes(5)


async def project(c, task="text_classification"):
    return (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]


async def make_class(c, pid, name="cat"):
    return (await c.post(f"/api/projects/{pid}/classes", json={"name": name})).json()["class"]["classId"]


def text_item(text, split="train", cls=None, external_id=None):
    item = {"split": split, "textFeatures": {"rawText": text}}
    if external_id:
        item["externalId"] = external_id
    if cls:
        item["annotations"] = [{"annotationType": "classification", "classId": cls}]
    return item


async def add(c, pid, *items):
    r = await c.post(f"/api/projects/{pid}/items", json={"items": list(items)})
    assert r.status_code == 200, r.text
    return r.json()


async def listing(c, pid, **params):
    r = await c.get(f"/api/projects/{pid}/items", params=params)
    assert r.status_code == 200, r.text
    return r.json()


# -- Creating and deduplicating items --------------------------------------------------------


async def test_creating_items_returns_them_with_features_annotations_and_draft_membership(client, new_user, db):
    c = await new_user()
    p = await project(c)
    cat = await make_class(c, p["id"])
    out = await add(c, p["id"], text_item("meow", "train", cat, external_id="a.txt"), text_item("woof", "test"))
    assert out["failed"] == [] and [i["externalId"] for i in out["created"]] == ["a.txt", None]

    page = await listing(c, p["id"], sort="oldest")
    first = page["items"][0]
    assert first["textFeatures"]["rawText"] == "meow" and first["splitType"] == "train"
    assert [(a["annotationType"], a["classId"]) for a in first["annotations"]] == [("classification", cat)]
    assert page["items"][1]["splitType"] == "test" and page["items"][1]["annotations"] == []


async def test_identical_content_is_deduplicated_within_and_across_batches_without_duplicating_labels(
    client, new_user, db
):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    first = await add(c, p["id"], text_item("same", "train", cat), text_item("same", "train", dog))
    assert first["failed"] == [] and first["created"][0]["id"] == first["created"][1]["id"]  # one pool item

    again = await add(c, p["id"], text_item("same", "validation", dog))
    assert again["created"][0]["id"] == first["created"][0]["id"]
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetItem))).scalar_one() == 1
        assert (await s.execute(sa.select(sa.func.count()).select_from(TextFeatures))).scalar_one() == 1
        anns = (await s.execute(sa.select(Annotation.class_id))).scalars().all()
    assert [str(a) for a in anns] == [cat]  # a dedup hit never re-labels content that is already in the pool

    only = (await listing(c, p["id"]))["items"]
    assert (
        len(only) == 1 and only[0]["splitType"] == "train"
    )  # membership kept its first split (do nothing on conflict)


async def test_tabular_dedup_ignores_key_order(client, new_user, db):
    c = await new_user()
    p = await project(c, "tabular_classification")
    a = {"split": "train", "tabularFeatures": {"featuresJson": {"age": 30, "city": "Hue", "nested": {"a": 1, "b": 2}}}}
    b = {"split": "train", "tabularFeatures": {"featuresJson": {"nested": {"b": 2, "a": 1}, "city": "Hue", "age": 30}}}
    out = await add(c, p["id"], a, b)
    assert out["created"][0]["id"] == out["created"][1]["id"]
    assert out["created"][0]["contentHash"] and len(out["created"][0]["contentHash"]) == 64


async def test_a_class_from_another_project_or_that_does_not_exist_is_rejected_before_anything_is_written(
    client, new_user, db
):
    c = await new_user()
    mine, other = await project(c), await project(c)
    foreign = await make_class(c, other["id"], "elsewhere")
    for bad in (foreign, str(uuid.uuid4())):
        r = await c.post(f"/api/projects/{mine['id']}/items", json={"items": [text_item("x", cls=bad)]})
        assert r.status_code == 400 and "Unknown label class" in r.json()["error"]["message"]
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetItem))).scalar_one() == 0


async def test_one_bad_item_fails_alone_and_leaves_no_orphan_rows(client, new_user, db):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    two_labels = {"split": "train", "textFeatures": {"rawText": "conflicted"}, "annotations": [
        {"annotationType": "classification", "classId": cat}, {"annotationType": "classification", "classId": dog}]}  # fmt: skip  # noqa: E501
    out = await add(c, p["id"], text_item("fine one"), two_labels, text_item("fine two"))
    assert [f["index"] for f in out["failed"]] == [1]
    assert "raw" not in out["failed"][0]["message"].lower() and "constraint" not in out["failed"][0]["message"].lower()
    assert len(out["created"]) == 2
    async with db() as s:
        texts = sorted((await s.execute(sa.select(TextFeatures.raw_text))).scalars().all())
        items = (await s.execute(sa.select(sa.func.count()).select_from(DatasetItem))).scalar_one()
    assert texts == ["fine one", "fine two"] and items == 2  # the failed item's row was rolled back with its features


async def test_item_body_validation(client, new_user):
    c = await new_user()
    p = await project(c)
    url = f"/api/projects/{p['id']}/items"
    assert (await c.post(url, json={"items": []})).status_code == 422
    assert (await c.post(url, json={"items": [{"split": "nope", "textFeatures": {"rawText": "x"}}]})).status_code == 422
    bad_conf = {
        "split": "train",
        "textFeatures": {"rawText": "x"},
        "annotations": [{"annotationType": "classification", "confidenceScore": 1.5}],
    }
    assert (await c.post(url, json={"items": [bad_conf]})).status_code == 422
    # storageUrl is not an accepted field: a client can never point an item at another tenant object.
    r = await c.post(
        url, json={"items": [{"split": "train", "storageUrl": "pool/other/x.png", "textFeatures": {"rawText": "x"}}]}
    )
    assert r.status_code == 200 and r.json()["created"][0]["storageUrl"] is None


async def test_a_large_csv_style_batch_is_fast_enough(client, new_user, db):
    c = await new_user()
    p = await project(c, "tabular_classification")
    items = [{"split": "train", "tabularFeatures": {"featuresJson": {"a": i, "b": f"row-{i}"}}} for i in range(1500)]
    loop = asyncio.get_running_loop()
    start = loop.time()
    r = await c.post(f"/api/projects/{p['id']}/items", json={"items": items})
    elapsed = loop.time() - start
    assert r.status_code == 200 and len(r.json()["created"]) == 1500 and r.json()["failed"] == []
    assert elapsed < 10, f"1500 items took {elapsed:.1f}s: the bulk fast path regressed to per-item inserts"


# -- Listing ---------------------------------------------------------------------------------


async def seeded(c):
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    await add(
        c, p["id"],
        text_item("alpha", "train", cat, "a.txt"), text_item("beta", "train", cat, "b.txt"),
        text_item("gamma", "validation", dog, "c.txt"), text_item("delta", "test", None, "d_%x.txt"),
    )  # fmt: skip
    return p, cat, dog


async def test_listing_counts_labels_classes_and_unassigned(client, new_user):
    c = await new_user()
    p, cat, dog = await seeded(c)
    page = await listing(c, p["id"])
    assert (page["total"], page["labeledCount"], page["unassignedCount"], page["page"], page["perPage"]) == (
        4,
        3,
        1,
        1,
        30,
    )
    assert {(x["classId"], x["count"]) for x in page["classCounts"]} == {(cat, 2), (dog, 1)}


async def test_split_and_class_filters_narrow_the_page_but_class_counts_ignore_the_class_filter(client, new_user):
    c = await new_user()
    p, cat, dog = await seeded(c)
    train = await listing(c, p["id"], split="train")
    assert train["total"] == 2 and {i["splitType"] for i in train["items"]} == {"train"}
    only_dog = await listing(c, p["id"], classId=dog)
    assert only_dog["total"] == 1 and only_dog["items"][0]["textFeatures"]["rawText"] == "gamma"
    # the dropdown counts must not collapse to the selected class
    assert {(x["classId"], x["count"]) for x in only_dog["classCounts"]} == {(cat, 2), (dog, 1)}
    unassigned = await listing(c, p["id"], classId="unassigned")
    assert unassigned["total"] == 1 and unassigned["items"][0]["textFeatures"]["rawText"] == "delta"
    assert (await listing(c, p["id"], split="not-a-split"))["total"] == 4  # an unknown split is ignored, not an error
    assert (await c.get(f"/api/projects/{p['id']}/items", params={"classId": "garbage"})).status_code == 400


async def test_search_matches_filename_case_insensitively_and_treats_wildcards_literally(client, new_user):
    c = await new_user()
    p, *_ = await seeded(c)
    assert [i["externalId"] for i in (await listing(c, p["id"], search="B.TXT"))["items"]] == ["b.txt"]
    assert [i["externalId"] for i in (await listing(c, p["id"], search="%"))["items"]] == [
        "d_%x.txt"
    ]  # not match-everything
    assert (await listing(c, p["id"], search="_"))["total"] == 1
    assert (await listing(c, p["id"], search="zzz"))["total"] == 0


async def test_sorting_and_pagination(client, new_user):
    c = await new_user()
    p, *_ = await seeded(c)
    ext = lambda **kw: [i["externalId"] for i in kw["page"]["items"]]  # noqa: E731
    assert ext(page=await listing(c, p["id"], sort="filename")) == ["a.txt", "b.txt", "c.txt", "d_%x.txt"]
    assert ext(page=await listing(c, p["id"], sort="oldest"))[0] == "a.txt"
    assert ext(page=await listing(c, p["id"]))[0] == "d_%x.txt"  # newest first is the default
    second = await listing(c, p["id"], sort="filename", perPage=2, page=2)
    assert ext(page=second) == ["c.txt", "d_%x.txt"] and second["total"] == 4
    assert (await c.get(f"/api/projects/{p['id']}/items", params={"perPage": 1001})).status_code == 422
    assert (await c.get(f"/api/projects/{p['id']}/items", params={"page": 0})).status_code == 422


async def test_listing_a_snapshot_or_a_foreign_version_and_ownership(client, new_user, db):
    owner, stranger = await new_user(), await new_user()
    p, *_ = await seeded(owner)
    other = await project(owner)
    async with db() as s:
        foreign = DatasetVersion(dataset_id=uuid.UUID(other["id"]), version_tag="x", status="ready")
        s.add(foreign)
        await s.commit()
    assert (await owner.get(f"/api/projects/{p['id']}/items", params={"versionId": str(foreign.id)})).status_code == 404
    assert (
        await owner.get(f"/api/projects/{p['id']}/items", params={"versionId": str(uuid.uuid4())})
    ).status_code == 404
    assert (await stranger.get(f"/api/projects/{p['id']}/items")).status_code == 403


# -- Upload ----------------------------------------------------------------------------------


async def test_uploading_an_image_stores_it_content_addressed_reads_its_size_and_labels_it(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p = await project(c, "image_classification")
    cat = await make_class(c, p["id"])
    r = await c.post(
        f"/api/projects/{p['id']}/upload", data={"split": "validation", "classId": cat},
        files=[("files", ("cat.png", png(64, 32), "image/png"))],
    )  # fmt: skip
    assert r.status_code == 200, r.text
    (res,) = r.json()["results"]
    assert res["status"] == "fulfilled" and res["value"]["isDuplicate"] is False
    value = res["value"]
    assert (
        value["externalId"] == "cat.png"
        and value["storageUrl"].startswith(f"pool/{p['id']}/")
        and value["storageUrl"].endswith(".png")
    )
    assert fake_s3[("theseus-datasets", value["storageUrl"])] == png(64, 32)

    item = (await listing(c, p["id"]))["items"][0]
    assert (
        item["visionFeatures"]["width"],
        item["visionFeatures"]["height"],
        item["visionFeatures"]["imageFormat"],
    ) == (64, 32, "png")
    assert item["splitType"] == "validation" and item["annotations"][0]["classId"] == cat
    assert item["downloadUrl"] == f"https://s3.test/theseus-datasets/{value['storageUrl']}"


async def test_re_uploading_identical_bytes_is_a_duplicate_that_is_not_relabeled_and_not_re_stored(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p = await project(c, "image_classification")
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    url = f"/api/projects/{p['id']}/upload"
    first = (
        await c.post(url, data={"split": "train", "classId": cat}, files=[("files", ("a.png", png(8, 8), "image/png"))])
    ).json()
    assert len(fake_s3.puts) == 1
    second = (
        await c.post(
            url, data={"split": "test", "classId": dog}, files=[("files", ("renamed.png", png(8, 8), "image/png"))]
        )
    ).json()

    v1, v2 = first["results"][0]["value"], second["results"][0]["value"]
    assert v2["id"] == v1["id"] and v2["isDuplicate"] is True
    assert len(fake_s3.puts) == 1  # the object already existed: the second upload wrote nothing
    item = (await listing(c, p["id"]))["items"][0]
    assert item["annotations"][0]["classId"] == cat and item["splitType"] == "train"  # neither label nor split changed


async def test_a_mixed_batch_reports_each_file_and_only_the_failing_one_fails(client, new_user, db, monkeypatch):
    monkeypatch.setattr(datasets_router, "MAX_UPLOAD_BYTES", 100)
    c = await new_user()
    p = await project(c, "image_classification")
    r = await c.post(
        f"/api/projects/{p['id']}/upload", data={"split": "train"},
        files=[("files", ("ok.png", png(4, 4), "image/png")), ("files", ("big.png", b"x" * 101, "image/png"))],
    )  # fmt: skip
    ok, big = r.json()["results"]
    assert ok["status"] == "fulfilled" and big["status"] == "rejected" and "50 MB" in big["reason"]
    assert (await listing(c, p["id"]))["total"] == 1


async def test_upload_validation(client, new_user):
    c = await new_user()
    p = await project(c, "image_classification")
    url = f"/api/projects/{p['id']}/upload"
    good = [("files", ("a.png", png(4, 4), "image/png"))]
    assert (await c.post(url, data={"split": "bogus"}, files=good)).status_code == 422
    assert (await c.post(url, data={"split": "train", "classId": str(uuid.uuid4())}, files=good)).status_code == 400
    assert (await c.post(url, data={"split": "train"})).status_code == 422  # files required


async def test_a_non_image_in_a_vision_project_is_stored_but_gets_no_vision_features(client, new_user, db):
    c = await new_user()
    p = await project(c, "image_classification")
    await c.post(
        f"/api/projects/{p['id']}/upload",
        data={"split": "train"},
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(VisionFeatures))).scalar_one() == 0
        assert (await s.execute(sa.select(sa.func.count()).select_from(DatasetItem))).scalar_one() == 1


# -- Snapshot versions -----------------------------------------------------------------------


async def wait_ready(c, version_id, timeout=15):
    for _ in range(int(timeout / 0.1)):
        v = (await c.get(f"/api/versions/{version_id}")).json()["version"]
        if v["status"] != "building":
            return v
        await asyncio.sleep(0.1)
    raise AssertionError("snapshot never left `building`")


async def test_snapshotting_copies_membership_and_builds_in_the_background(client, new_user, db, fake_s3):
    c = await new_user()
    p, cat, dog = await seeded(c)
    r = await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v1"})
    assert r.status_code == 202
    created = r.json()["version"]
    assert (created["versionTag"], created["status"], created["itemCount"]) == ("v1", "building", 4)

    v = await wait_ready(c, created["id"])
    assert v["status"] == "ready" and v["itemCount"] == 4 and v["classCount"] == 2
    assert {s["splitType"]: s["itemCount"] for s in v["splits"]} == {"train": 2, "validation": 1, "test": 1}
    assert v["dataset"]["modality"] == "text" and v["parquetKey"] == f"snapshots/{created['id']}/dataset.parquet"
    assert ("theseus-datasets", v["parquetKey"]) in fake_s3

    # The draft keeps evolving; the snapshot is immutable.
    await add(c, p["id"], text_item("epsilon"))
    assert (await listing(c, p["id"]))["total"] == 5
    assert (await listing(c, p["id"], versionId=created["id"]))["total"] == 4


async def test_snapshot_tags_are_unique_and_validated(client, new_user):
    c = await new_user()
    p = await project(c)
    url = f"/api/projects/{p['id']}/versions"
    assert (await c.post(url, json={"versionTag": "v1"})).status_code == 202
    dup = await c.post(url, json={"versionTag": "v1"})
    assert dup.status_code == 409 and "already exists" in dup.json()["error"]["message"]
    assert (await c.post(url, json={"versionTag": ""})).status_code == 422
    assert (await c.post(url, json={"versionTag": "x" * 51})).status_code == 422


async def test_the_draft_cannot_be_deleted_but_a_snapshot_can_and_its_objects_are_removed(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p, *_ = await seeded(c)
    detail = (await c.get(f"/api/projects/{p['id']}")).json()["project"]["dataset"]
    assert (await c.delete(f"/api/versions/{detail['draft']['id']}")).status_code == 400

    vid = (await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v1"})).json()["version"]["id"]
    await wait_ready(c, vid)
    assert (await c.delete(f"/api/versions/{vid}")).status_code == 204
    assert (await c.get(f"/api/versions/{vid}")).status_code == 404
    assert ("theseus-datasets", f"snapshots/{vid}/dataset.parquet") in fake_s3.deleted


async def test_version_routes_are_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    vid = (await owner.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v1"})).json()["version"]["id"]
    assert (await stranger.get(f"/api/versions/{vid}")).status_code == 403
    assert (await stranger.delete(f"/api/versions/{vid}")).status_code == 403
    assert (await stranger.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v2"})).status_code == 403


# -- Deleting items --------------------------------------------------------------------------


async def bulk_delete(c, pid, ids):
    return await c.request("DELETE", f"/api/projects/{pid}/items", json={"itemIds": ids})


async def test_deleting_a_pool_only_item_is_a_hard_delete_and_removes_its_file(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c, "image_classification")
    res = (
        await c.post(
            f"/api/projects/{p['id']}/upload",
            data={"split": "train"},
            files=[("files", ("a.png", png(4, 4), "image/png"))],
        )
    ).json()
    item = res["results"][0]["value"]
    r = await bulk_delete(c, p["id"], [item["id"]])
    assert r.json()["results"] == [{"itemId": item["id"], "outcome": "deleted"}]
    assert ("theseus-datasets", item["storageUrl"]) in fake_s3.deleted
    assert (await listing(c, p["id"]))["total"] == 0


async def test_an_item_referenced_by_a_snapshot_is_soft_deleted_so_the_snapshot_stays_immutable(client, new_user, db):
    c = await new_user()
    p, *_ = await seeded(c)
    vid = (await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v1"})).json()["version"]["id"]
    await wait_ready(c, vid)
    victim = (await listing(c, p["id"], search="a.txt"))["items"][0]["id"]

    r = await bulk_delete(c, p["id"], [victim])
    assert r.json()["results"] == [{"itemId": victim, "outcome": "soft_deleted"}]
    assert (await listing(c, p["id"]))["total"] == 3  # gone from the draft
    snap = await listing(c, p["id"], versionId=vid)
    assert snap["total"] == 4 and any(
        i["id"] == victim and i["deletedAt"] for i in snap["items"]
    )  # still in the snapshot
    async with db() as s:
        assert (
            await s.execute(sa.select(sa.func.count()).select_from(TextFeatures))
        ).scalar_one() == 4  # features intact


async def test_re_adding_soft_deleted_content_restores_the_item(client, new_user):
    c = await new_user()
    p = await project(c)
    out = await add(c, p["id"], text_item("comeback", external_id="x"))
    vid = (await c.post(f"/api/projects/{p['id']}/versions", json={"versionTag": "v1"})).json()["version"]["id"]
    await wait_ready(c, vid)
    await bulk_delete(c, p["id"], [out["created"][0]["id"]])
    assert (await listing(c, p["id"]))["total"] == 0

    again = await add(c, p["id"], text_item("comeback"))
    assert again["created"][0]["id"] == out["created"][0]["id"] and again["created"][0]["deletedAt"] is None
    assert (await listing(c, p["id"]))["total"] == 1


async def test_delete_reports_not_found_for_ids_outside_the_project_and_the_single_route_works(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p, other = await project(owner), await project(owner)
    mine = (await add(owner, p["id"], text_item("mine")))["created"][0]["id"]
    theirs = (await add(owner, other["id"], text_item("theirs")))["created"][0]["id"]

    r = await bulk_delete(owner, p["id"], [theirs, str(uuid.uuid4())])
    assert [x["outcome"] for x in r.json()["results"]] == [
        "not_found",
        "not_found",
    ]  # another project's item is untouchable
    assert (await listing(owner, other["id"]))["total"] == 1

    assert (await stranger.delete(f"/api/items/{mine}")).status_code == 403
    assert (await bulk_delete(stranger, p["id"], [mine])).status_code == 403
    assert (await owner.delete(f"/api/items/{mine}")).status_code == 204
    assert (await bulk_delete(owner, p["id"], [])).status_code == 422
