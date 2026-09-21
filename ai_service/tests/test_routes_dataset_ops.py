"""Dataset operations over HTTP: splits, labeling, auto-split, health, annotations; plus the pure split planner."""

import asyncio
import random
import struct
import uuid

import pytest
import sqlalchemy as sa

from theseus.db.models import Annotation
from theseus.services import datasets as svc
from theseus.services import storage


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    store: dict[tuple[str, str], bytes] = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__((b, k), data))
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    for name in ("delete_file", "delete_files", "delete_prefix"):
        monkeypatch.setattr(storage, name, lambda *a, **k: None)
    return store


def png(w, h, extra=b""):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + bytes(5) + extra


async def project(c, task="text_classification"):
    return (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]


async def make_class(c, pid, name):
    return (await c.post(f"/api/projects/{pid}/classes", json={"name": name})).json()["class"]["classId"]


def item(text, split="train", cls=None, **extra):
    body = {"split": split, "textFeatures": {"rawText": text, **extra}}
    if cls:
        body["annotations"] = [{"annotationType": "classification", "classId": cls}]
    return body


async def add(c, pid, *items):
    r = await c.post(f"/api/projects/{pid}/items", json={"items": list(items)})
    assert r.status_code == 200, r.text
    return [x["id"] for x in r.json()["created"]]


async def page(c, pid, **params):
    return (await c.get(f"/api/projects/{pid}/items", params={"perPage": 1000, **params})).json()


# -- Bulk split ------------------------------------------------------------------------------


async def test_setting_the_split_touches_only_the_named_draft_items(client, new_user):
    c = await new_user()
    p = await project(c)
    a, b, keep = await add(c, p["id"], item("a"), item("b"), item("c"))
    r = await c.patch(
        f"/api/projects/{p['id']}/items/split", json={"itemIds": [a, b, str(uuid.uuid4())], "split": "test"}
    )
    assert sorted(r.json()["updated"]) == sorted([a, b])  # the unknown id is simply not in the draft
    splits = {i["id"]: i["splitType"] for i in (await page(c, p["id"]))["items"]}
    assert splits == {a: "test", b: "test", keep: "train"}


async def test_split_validation_and_ownership(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    (a,) = await add(owner, p["id"], item("a"))
    url = f"/api/projects/{p['id']}/items/split"
    assert (await owner.patch(url, json={"itemIds": [a], "split": "bogus"})).status_code == 422
    assert (await owner.patch(url, json={"itemIds": [], "split": "test"})).status_code == 422
    assert (await stranger.patch(url, json={"itemIds": [a], "split": "test"})).status_code == 403


# -- Classify --------------------------------------------------------------------------------


async def test_classify_creates_then_updates_a_single_classification_per_item(client, new_user, db):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    a, b = await add(c, p["id"], item("a"), item("b", cls=cat))
    url = f"/api/projects/{p['id']}/items/classify"

    assert (await c.post(url, json={"itemIds": [a, b], "classId": dog})).json() == {"updated": 2, "failed": 0}
    by_id = {i["id"]: i["annotations"] for i in (await page(c, p["id"]))["items"]}
    assert [x["classId"] for x in by_id[a]] == [dog] and [x["classId"] for x in by_id[b]] == [
        dog
    ]  # b was relabeled, not doubled
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(Annotation))).scalar_one() == 2


async def test_classify_rejects_a_foreign_class_and_counts_foreign_items_as_failed(client, new_user):
    c = await new_user()
    mine, other = await project(c), await project(c)
    cat, elsewhere = await make_class(c, mine["id"], "cat"), await make_class(c, other["id"], "x")
    (m,) = await add(c, mine["id"], item("mine"))
    (o,) = await add(c, other["id"], item("theirs"))
    url = f"/api/projects/{mine['id']}/items/classify"
    assert (await c.post(url, json={"itemIds": [m], "classId": elsewhere})).status_code == 400
    assert (await c.post(url, json={"itemIds": [m, o, str(uuid.uuid4())], "classId": cat})).json() == {
        "updated": 1,
        "failed": 2,
    }
    assert (await page(c, other["id"]))["items"][0]["annotations"] == []  # another project's item was never labeled


async def test_concurrent_classify_calls_can_never_leave_two_classifications_on_one_item(client, new_user, db):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    ids = await add(c, p["id"], *[item(f"row {i}") for i in range(20)])
    url = f"/api/projects/{p['id']}/items/classify"
    results = await asyncio.gather(
        *(c.post(url, json={"itemIds": ids, "classId": cat if n % 2 else dog}) for n in range(6))
    )
    assert all(r.status_code == 200 for r in results)
    async with db() as s:
        per_item = (
            (await s.execute(sa.select(sa.func.count()).select_from(Annotation).group_by(Annotation.item_id)))
            .scalars()
            .all()
        )
    assert (
        per_item == [1] * 20
    )  # the partial unique index plus a single upsert statement close the read-then-write race


# -- Auto-split ------------------------------------------------------------------------------


async def test_auto_split_assigns_every_item_exactly_once_and_reports_the_counts(client, new_user):
    c = await new_user()
    p = await project(c, "tabular_regression")
    await add(c, p["id"], *[{"split": "train", "tabularFeatures": {"featuresJson": {"x": i}}} for i in range(100)])
    r = (await c.post(f"/api/projects/{p['id']}/items/auto-split", json={})).json()
    assert (r["updated"], r["stratified"], r["warnings"]) == (
        100,
        False,
        [],
    )  # regression has no classes to stratify by
    assert r["splits"] == {"train": 80, "validation": 10, "test": 10}
    counts = {}
    for i in (await page(c, p["id"]))["items"]:
        counts[i["splitType"]] = counts.get(i["splitType"], 0) + 1
    assert counts == {"train": 80, "validation": 10, "test": 10}


async def test_a_stratified_split_keeps_every_class_in_every_split_and_warns_about_tiny_classes(client, new_user):
    c = await new_user()
    p = await project(c)
    big, tiny = await make_class(c, p["id"], "big"), await make_class(c, p["id"], "tiny")
    await add(c, p["id"], *[item(f"big {i}", cls=big) for i in range(30)], item("t1", cls=tiny), item("t2", cls=tiny))
    r = (
        await c.post(
            f"/api/projects/{p['id']}/items/auto-split", json={"ratios": {"train": 2, "validation": 1, "test": 1}}
        )
    ).json()
    assert r["stratified"] is True and r["updated"] == 32
    assert len(r["warnings"]) == 1 and tiny in r["warnings"][0] and "2 item(s)" in r["warnings"][0]

    per = {}
    for i in (await page(c, p["id"]))["items"]:
        per.setdefault(i["annotations"][0]["classId"], set()).add(i["splitType"])
    assert per[big] == {"train", "validation", "test"}  # a big class cannot be shuffled out of any split


async def test_auto_split_edge_cases(client, new_user):
    c = await new_user()
    p = await project(c)
    url = f"/api/projects/{p['id']}/items/auto-split"
    assert (await c.post(url, json={})).json() == {"updated": 0}  # nothing to split
    await add(c, p["id"], item("a"), item("b"))
    assert (await c.post(url, json={"ratios": {"train": 0, "validation": 0, "test": 0}})).status_code == 400
    assert (await c.post(url, json={"ratios": {"train": -1, "validation": 1, "test": 1}})).status_code == 422
    only_test = (await c.post(url, json={"ratios": {"train": 0, "validation": 0, "test": 5}, "stratify": False})).json()
    assert only_test["splits"] == {"train": 0, "validation": 0, "test": 2}


def test_plan_split_assigns_everything_once_with_test_absorbing_rounding():
    ids = [uuid.uuid4() for _ in range(7)]
    plan = svc.plan_split([(i, "all") for i in ids], {"train": 1, "validation": 1, "test": 1}, False, random.Random(1))
    flat = [i for g in plan.groups.values() for i in g]
    assert sorted(flat) == sorted(ids) and len(flat) == 7  # exactly once, none lost to rounding
    assert [len(plan.groups[s]) for s in ("train", "validation", "test")] == [2, 2, 3]


def test_plan_split_is_reproducible_with_a_seed_and_stratifies_per_class():
    members = [(uuid.uuid4(), "a") for _ in range(10)] + [(uuid.uuid4(), "b") for _ in range(10)]
    ratios = {"train": 60, "validation": 20, "test": 20}
    p1 = svc.plan_split(members, ratios, True, random.Random(7))
    p2 = svc.plan_split([(i, k) for i, k in members], ratios, True, random.Random(7))
    assert p1.groups == p2.groups
    klass = dict(members)
    for split in ("train", "validation", "test"):
        assert {klass[i] for i in p1.groups[split]} == {"a", "b"}  # both classes present in every split
    with pytest.raises(ValueError):
        svc.plan_split(members, {"train": 0, "validation": 0, "test": 0}, False)


def test_js_round_matches_javascript_for_halves():
    assert [svc.js_round(x) for x in (0.5, 1.5, 2.5, 3.5, 0.49)] == [
        1,
        2,
        3,
        4,
        0,
    ]  # Python round() would give 0, 2, 2, 4, 0


# -- Health ----------------------------------------------------------------------------------


async def test_health_of_an_empty_draft(client, new_user):
    c = await new_user()
    p = await project(c)
    h = (await c.get(f"/api/projects/{p['id']}/dataset/health")).json()["health"]
    assert (h["itemCount"], h["labeledCount"], h["modality"], h["classDistribution"], h["text"]) == (
        0,
        0,
        "text",
        [],
        None,
    )


async def test_text_health_reports_labels_classes_tokens_and_languages(client, new_user):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    await add(
        c, p["id"],
        item("a", cls=cat, tokenCount=10, languageCode="en"), item("b", cls=cat, tokenCount=20, languageCode="en"),
        item("c", cls=cat, tokenCount=30), item("d", cls=dog, tokenCount=41, languageCode="vi"), item("e"),
    )  # fmt: skip
    h = (await c.get(f"/api/projects/{p['id']}/dataset/health")).json()["health"]
    assert (h["itemCount"], h["labeledCount"], h["unlabeledCount"]) == (5, 4, 1)
    assert [(x["name"], x["count"]) for x in h["classDistribution"]] == [("cat", 3), ("dog", 1)]  # most frequent first
    assert [x["name"] for x in h["smallClasses"]] == ["dog"]  # under the 3-item threshold
    assert h["text"]["count"] == 5 and h["text"]["languages"] == {"en": 2, "vi": 1, "unknown": 2}
    assert h["text"]["tokenCount"] == {
        "min": 10.0,
        "max": 41.0,
        "avg": 25,
    }  # avg of 4 counted rows, rounded like Math.round
    assert (h["duplicateContentHashes"], h["missingContentHash"]) == (0, 0)
    assert h["vision"] is None and h["audio"] is None and h["tabular"] is None


async def test_vision_health_reads_dimensions_and_formats_from_uploads(client, new_user):
    c = await new_user()
    p = await project(c, "image_classification")
    files = [
        ("files", (f"{n}.png", png(w, h, bytes([n])), "image/png")) for n, (w, h) in enumerate([(100, 50), (300, 150)])
    ]
    await c.post(f"/api/projects/{p['id']}/upload", data={"split": "train"}, files=files)
    h = (await c.get(f"/api/projects/{p['id']}/dataset/health")).json()["health"]
    assert h["vision"] == {"count": 2, "width": {"min": 100.0, "max": 300.0, "avg": 200.0},
                           "height": {"min": 50.0, "max": 150.0, "avg": 100.0}, "formats": {"png": 2}}  # fmt: skip


async def test_audio_and_tabular_health(client, new_user):
    c = await new_user()
    a = await project(c, "audio_classification")
    audio = lambda d, r: {  # noqa: E731
        "split": "train",
        "audioFeatures": {"durationSeconds": d, "sampleRateHz": r, "audioCodec": "wav"},
    }  # noqa: E731
    await add(c, a["id"], audio(1.5, 16000), audio(2.5, 16000), audio(5.0, 44100))
    h = (await c.get(f"/api/projects/{a['id']}/dataset/health")).json()["health"]["audio"]
    assert h["count"] == 3 and h["sampleRates"] == {"16000": 2, "44100": 1}
    assert h["durationSeconds"] == {"min": 1.5, "max": 5.0, "avg": pytest.approx(3.0)}

    t = await project(c, "tabular_classification")
    await add(c, t["id"], *[{"split": "train", "tabularFeatures": {"featuresJson": {"x": i}}} for i in range(4)])
    assert (await c.get(f"/api/projects/{t['id']}/dataset/health")).json()["health"]["tabular"] == {"count": 4}


async def test_health_is_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    assert (await stranger.get(f"/api/projects/{p['id']}/dataset/health")).status_code == 403


# -- Annotations -----------------------------------------------------------------------------


async def test_annotation_lifecycle(client, new_user):
    c = await new_user()
    p = await project(c)
    cat, dog = await make_class(c, p["id"], "cat"), await make_class(c, p["id"], "dog")
    (a,) = await add(c, p["id"], item("a"))

    created = await c.post(
        f"/api/items/{a}/annotations",
        json={"annotationType": "classification", "classId": cat, "confidenceScore": 0.75},
    )
    assert created.status_code == 200
    ann = created.json()["annotation"]
    assert (ann["itemId"], ann["classId"], ann["annotationType"], ann["confidenceScore"]) == (
        a,
        cat,
        "classification",
        "0.750",
    )
    assert [x["id"] for x in (await c.get(f"/api/items/{a}/annotations")).json()["annotations"]] == [ann["id"]]

    patched = (await c.patch(f"/api/annotations/{ann['id']}", json={"classId": dog})).json()["annotation"]
    assert (
        patched["classId"] == dog and patched["confidenceScore"] == "0.750"
    )  # fields that were not sent are untouched
    caption = (
        await c.post(
            f"/api/items/{a}/annotations", json={"annotationType": "text_sequence", "labelTextSequence": "a cat"}
        )
    ).json()["annotation"]
    edited = (await c.patch(f"/api/annotations/{caption['id']}", json={"labelTextSequence": "a dog"})).json()[
        "annotation"
    ]
    assert edited["labelTextSequence"] == "a dog"

    assert (await c.delete(f"/api/annotations/{ann['id']}")).status_code == 204
    assert [x["id"] for x in (await c.get(f"/api/items/{a}/annotations")).json()["annotations"]] == [caption["id"]]
    assert (await c.delete(f"/api/annotations/{ann['id']}")).status_code == 404


async def test_annotation_validation_and_the_one_classification_rule(client, new_user):
    c = await new_user()
    mine, other = await project(c), await project(c)
    cat, elsewhere = await make_class(c, mine["id"], "cat"), await make_class(c, other["id"], "x")
    (a,) = await add(c, mine["id"], item("a"))
    url = f"/api/items/{a}/annotations"
    assert (await c.post(url, json={"annotationType": "classification", "classId": elsewhere})).status_code == 400
    assert (await c.post(url, json={"annotationType": "classification", "confidenceScore": 2})).status_code == 422
    assert (await c.post(url, json={"annotationType": "nonsense"})).status_code == 422
    assert (await c.post(url, json={"annotationType": "classification", "classId": cat})).status_code == 200
    dup = await c.post(url, json={"annotationType": "classification", "classId": cat})
    assert dup.status_code == 409 and "already has a classification" in dup.json()["error"]["message"]
    ann = (await c.get(url)).json()["annotations"][0]
    assert (await c.patch(f"/api/annotations/{ann['id']}", json={"classId": elsewhere})).status_code == 400


async def test_annotation_routes_are_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    (a,) = await add(owner, p["id"], item("a"))
    ann = (
        await owner.post(
            f"/api/items/{a}/annotations", json={"annotationType": "text_sequence", "labelTextSequence": "x"}
        )
    ).json()["annotation"]
    for call in (
        stranger.get(f"/api/items/{a}/annotations"),
        stranger.post(f"/api/items/{a}/annotations", json={"annotationType": "text_sequence"}),
        stranger.patch(f"/api/annotations/{ann['id']}", json={"labelTextSequence": "y"}),
        stranger.delete(f"/api/annotations/{ann['id']}"),
    ):
        assert (await call).status_code == 403
    assert (await owner.get(f"/api/items/{uuid.uuid4()}/annotations")).status_code == 404


async def test_classifying_many_items_in_one_request_all_succeed(client, new_user):
    """Regression: after 5 executions Postgres switches a prepared statement to a generic plan. An
    ON CONFLICT ... WHERE predicate sent as a bind parameter then cannot be matched to the partial
    unique index, so every item past the fifth used to fail."""
    c = await new_user()
    p = await project(c)
    cat = await make_class(c, p["id"], "cat")
    ids = await add(c, p["id"], *[item(f"row {i}") for i in range(40)])
    r = await c.post(f"/api/projects/{p['id']}/items/classify", json={"itemIds": ids, "classId": cat})
    assert r.json() == {"updated": 40, "failed": 0}
    assert all(len(i["annotations"]) == 1 for i in (await page(c, p["id"]))["items"])
