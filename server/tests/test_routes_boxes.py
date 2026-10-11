"""Bounding boxes over HTTP: one annotation per box, the "no objects" marker, per-class counts and filters,
merging classes, and the endpoints that must refuse boxes."""

import pytest

BOX = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
MARKER = {"background": True}


async def project(c, task="object_detection"):
    r = await c.post("/api/projects", json={"name": "p", "description": None, "task": task})
    assert r.status_code == 200, r.text
    return r.json()["project"]


async def make_class(c, pid, name):
    r = await c.post(f"/api/projects/{pid}/classes", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()["class"]["classId"]


async def add_item(c, pid, name, split="train"):
    body = {"items": [{"split": split, "externalId": name, "visionFeatures": {"width": 640, "height": 480}}]}
    r = await c.post(f"/api/projects/{pid}/items", json=body)
    assert r.status_code == 200, r.text
    created = r.json()["created"]
    assert len(created) == 1, r.text
    return created[0]["id"]


async def annotate(c, item_id, class_id=None, label=BOX, kind="bounding_box"):
    body = {"annotationType": kind, "labelStructured": label}
    if class_id:
        body["classId"] = class_id
    return await c.post(f"/api/items/{item_id}/annotations", json=body)


async def box(c, item_id, class_id, label=BOX):
    r = await annotate(c, item_id, class_id, label)
    assert r.status_code == 200, r.text
    return r.json()["annotation"]


async def annotations(c, item_id):
    r = await c.get(f"/api/items/{item_id}/annotations")
    assert r.status_code == 200, r.text
    return r.json()["annotations"]


async def listing(c, pid, **params):
    r = await c.get(f"/api/projects/{pid}/items", params=params)
    assert r.status_code == 200, r.text
    return r.json()


async def stats(c, pid):
    r = await c.get(f"/api/projects/{pid}/classes/stats")
    assert r.status_code == 200, r.text
    return r.json()


async def setup(c, *class_names):
    p = await project(c)
    return p["id"], [await make_class(c, p["id"], n) for n in class_names]


# -- Creating boxes --------------------------------------------------------------------------------------


async def test_a_box_is_one_annotation_with_a_class_and_normalised_coordinates(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")

    ann = await box(c, item, cat)
    assert (ann["annotationType"], ann["classId"], ann["labelStructured"]) == ("bounding_box", cat, BOX)


async def test_an_item_can_hold_many_boxes_of_several_classes(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    item = await add_item(c, pid, "a")

    await box(c, item, cat)
    await box(c, item, cat, {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2})
    await box(c, item, dog, {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1})

    assert [a["classId"] for a in await annotations(c, item)] == [cat, cat, dog]  # unlike classification, no cap of one


@pytest.mark.parametrize(
    "label",
    [
        {"x": 0.8, "y": 0.1, "w": 0.4, "h": 0.2},  # runs off the right edge
        {"x": -0.1, "y": 0.1, "w": 0.4, "h": 0.2},
        {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.2},
        {"x": 0.1, "y": 0.1, "w": 0.4},
        {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.2, "angle": 5},
        None,
    ],
)
async def test_a_malformed_box_is_rejected(client, new_user, label):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")

    assert (await annotate(c, item, cat, label)).status_code == 422
    assert await annotations(c, item) == []


async def test_a_box_needs_a_class_and_a_class_of_this_project(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    other_pid, (foreign,) = await setup(c, "foreign")
    item = await add_item(c, pid, "a")

    assert (await annotate(c, item, None, BOX)).status_code == 422  # no class
    assert (await annotate(c, item, foreign, BOX)).status_code == 400  # a class from another project's dataset
    assert other_pid != pid
    assert await annotations(c, item) == []


async def test_creating_items_with_an_inline_invalid_box_is_rejected(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    body = {
        "items": [
            {
                "split": "train",
                "externalId": "x",
                "visionFeatures": {"width": 10, "height": 10},
                "annotations": [{"annotationType": "bounding_box", "classId": cat, "labelStructured": {"x": 2}}],
            }
        ]
    }
    assert (await c.post(f"/api/projects/{pid}/items", json=body)).status_code == 422


# -- The "no objects" marker -----------------------------------------------------------------------------


async def test_a_marker_says_the_image_is_labelled_and_empty(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    marked, unlabelled = await add_item(c, pid, "marked"), await add_item(c, pid, "unlabelled")

    r = await annotate(c, marked, None, MARKER)
    assert r.status_code == 200, r.text
    assert r.json()["annotation"]["classId"] is None

    page = await listing(c, pid)
    assert page["labeledCount"] == 1  # the marked one counts as labelled, the other does not
    assert (await listing(c, pid, labeled="true"))["total"] == 1
    assert (await listing(c, pid, labeled="false"))["items"][0]["id"] == unlabelled


async def test_a_marker_with_a_class_is_rejected(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")
    assert (await annotate(c, item, cat, MARKER)).status_code == 422


async def test_drawing_a_box_on_a_marked_image_clears_the_marker(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")
    await annotate(c, item, None, MARKER)

    await box(c, item, cat)

    remaining = await annotations(c, item)
    assert [a["classId"] for a in remaining] == [cat]  # the marker's meaning ("nothing here") no longer holds


async def test_marking_an_image_that_has_boxes_is_refused_and_a_second_marker_too(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    boxed, empty = await add_item(c, pid, "boxed"), await add_item(c, pid, "empty")
    await box(c, boxed, cat)

    r = await annotate(c, boxed, None, MARKER)
    assert r.status_code == 409 and "delete them" in r.json()["error"]["message"]

    assert (await annotate(c, empty, None, MARKER)).status_code == 200
    assert (await annotate(c, empty, None, MARKER)).status_code == 409


# -- Editing and deleting a box --------------------------------------------------------------------------


async def test_a_box_can_be_moved_reclassed_and_deleted(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    item = await add_item(c, pid, "a")
    ann = await box(c, item, cat)

    moved = await c.patch(
        f"/api/annotations/{ann['id']}", json={"labelStructured": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}}
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["annotation"]["labelStructured"]["w"] == 0.5
    assert moved.json()["annotation"]["classId"] == cat  # only what was sent changed

    reclassed = await c.patch(f"/api/annotations/{ann['id']}", json={"classId": dog})
    assert reclassed.json()["annotation"]["classId"] == dog
    assert reclassed.json()["annotation"]["labelStructured"]["w"] == 0.5

    assert (await c.delete(f"/api/annotations/{ann['id']}")).status_code == 204
    assert await annotations(c, item) == []


async def test_an_update_cannot_make_a_box_invalid(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")
    ann = await box(c, item, cat)

    bad = await c.patch(
        f"/api/annotations/{ann['id']}", json={"labelStructured": {"x": 0.9, "y": 0.9, "w": 0.5, "h": 0.5}}
    )
    assert bad.status_code == 422
    assert (await annotations(c, item))[0]["labelStructured"] == BOX  # untouched


# -- Counts, filters, health -----------------------------------------------------------------------------


async def test_item_listing_counts_items_per_class_once_however_many_boxes(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    both, cats, nothing, marked = [await add_item(c, pid, n) for n in ("both", "cats", "nothing", "marked")]
    await box(c, both, cat)
    await box(c, both, cat)
    await box(c, both, dog)
    await box(c, cats, cat)
    await annotate(c, marked, None, MARKER)

    page = await listing(c, pid)
    counts = {x["classId"]: x["count"] for x in page["classCounts"]}
    assert counts == {cat: 2, dog: 1}  # `both` counts once for cat although it has two cat boxes
    assert page["unassignedCount"] == 1  # only `nothing`: the marked image is labelled, just with no class
    assert page["labeledCount"] == 3


async def test_filtering_by_class_means_has_a_box_of_that_class(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    both, cats, nothing = [await add_item(c, pid, n) for n in ("both", "cats", "nothing")]
    await box(c, both, cat)
    await box(c, both, dog)
    await box(c, cats, cat)

    ids = lambda page: {i["id"] for i in page["items"]}  # noqa: E731
    assert ids(await listing(c, pid, classId=cat)) == {both, cats}
    assert ids(await listing(c, pid, classId=dog)) == {both}
    assert ids(await listing(c, pid, classId="unassigned")) == {nothing}


async def test_class_stats_count_items_per_class_and_split_and_the_unlabelled(client, new_user):
    c = await new_user()
    pid, (cat, dog, empty_class) = await setup(c, "cat", "dog", "unused")
    train_both = await add_item(c, pid, "tb", "train")
    train_cat = await add_item(c, pid, "tc", "train")
    val_dog = await add_item(c, pid, "vd", "validation")
    await add_item(c, pid, "unlabelled", "test")
    marked = await add_item(c, pid, "marked", "test")
    await box(c, train_both, cat)
    await box(c, train_both, cat)
    await box(c, train_both, dog)
    await box(c, train_cat, cat)
    await box(c, val_dog, dog)
    await annotate(c, marked, None, MARKER)

    body = await stats(c, pid)
    by_id = {s["classId"]: s for s in body["stats"]}
    assert body["counted"] is True
    assert (by_id[cat]["total"], by_id[cat]["train"], by_id[cat]["validation"]) == (2, 2, 0)  # an item counts once
    assert (by_id[dog]["total"], by_id[dog]["train"], by_id[dog]["validation"]) == (2, 1, 1)
    assert by_id[empty_class]["total"] == 0
    assert body["total"] == 5  # items, not boxes and not class memberships
    assert body["unassigned"] == 1  # only the image nobody labelled


async def test_merging_classes_relabels_every_box(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    item = await add_item(c, pid, "a")
    await box(c, item, cat)
    await box(c, item, dog, {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2})

    r = await c.put(
        f"/api/projects/{pid}/classes",
        json={"classes": [{"classId": cat, "name": "cat"}], "merges": [{"fromClassId": dog, "intoClassId": cat}]},
    )
    assert r.status_code == 200, r.text
    assert [a["classId"] for a in await annotations(c, item)] == [cat, cat]  # no uniqueness to collide with


async def test_health_counts_labelled_images_and_images_per_class(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    a, b, _ = [await add_item(c, pid, n) for n in ("a", "b", "c")]
    await box(c, a, cat)
    await box(c, a, cat)
    await annotate(c, b, None, MARKER)

    r = await c.get(f"/api/projects/{pid}/dataset/health")
    assert r.status_code == 200, r.text
    health = r.json()["health"]
    assert (health["itemCount"], health["labeledCount"], health["unlabeledCount"]) == (3, 2, 1)
    assert [(d["name"], d["count"]) for d in health["classDistribution"]] == [("cat", 1)]  # images, not boxes


# -- Endpoints that must refuse boxes --------------------------------------------------------------------


async def test_bulk_classify_is_refused_because_a_class_belongs_to_one_box(client, new_user):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    item = await add_item(c, pid, "a")

    r = await c.post(f"/api/projects/{pid}/items/classify", json={"itemIds": [item], "classId": cat})
    assert r.status_code == 400 and "one at a time" in r.json()["error"]["message"]
    assert await annotations(c, item) == []


async def test_auto_split_works_with_several_boxes_per_item(client, new_user):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    for i in range(12):
        item = await add_item(c, pid, f"i{i}")
        await box(c, item, cat if i % 2 else dog)
        await box(c, item, dog)

    r = await c.post(f"/api/projects/{pid}/items/auto-split", json={})
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 12 and r.json()["stratified"] is True


# -- Uploading images together with their boxes (the YOLO-format import) ---------------------------------


@pytest.fixture
def fake_s3(monkeypatch):
    from omnitrain.services import storage

    store: dict = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__((b, k), data))
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    return store


def png(width: int, height: int, fill: int = 0) -> bytes:
    """The header of a PNG, enough for the dimension reader; `fill` makes the bytes (so the hash) differ."""
    import struct

    return (
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + bytes([fill]) * 5
    )


async def upload(c, pid, files, boxes=None, **form):
    import json

    data = {"split": "train", **form}
    if boxes is not None:
        data["boxes"] = json.dumps(boxes)
    return await c.post(
        f"/api/projects/{pid}/upload",
        data=data,
        files=[("files", (name, body, "image/png")) for name, body in files],
    )


def api_box(class_id, **overrides):
    return {"classId": class_id, **BOX, **overrides}


async def test_images_upload_with_their_boxes_a_marker_or_nothing(client, new_user, fake_s3):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")

    r = await upload(
        c,
        pid,
        [("boxed.png", png(64, 48, 1)), ("empty.png", png(64, 48, 2)), ("unlabelled.png", png(64, 48, 3))],
        boxes=[[api_box(cat), api_box(dog, x=0.5, y=0.5, w=0.2, h=0.2)], [], None],
    )
    assert r.status_code == 200, r.text
    assert [x["status"] for x in r.json()["results"]] == ["fulfilled"] * 3

    by_name = {i["externalId"]: i for i in (await listing(c, pid))["items"]}
    boxed = by_name["boxed.png"]["annotations"]
    assert [(a["annotationType"], a["classId"]) for a in boxed] == [("bounding_box", cat), ("bounding_box", dog)]
    assert boxed[0]["labelStructured"] == BOX
    (marker,) = by_name["empty.png"]["annotations"]
    assert (marker["classId"], marker["labelStructured"]) == (None, MARKER)  # labelled, with nothing in it
    assert by_name["unlabelled.png"]["annotations"] == []  # nobody labelled it


async def test_a_duplicate_upload_keeps_the_labels_it_already_has(client, new_user, fake_s3):
    c = await new_user()
    pid, (cat, dog) = await setup(c, "cat", "dog")
    image = png(64, 48, 9)
    await upload(c, pid, [("a.png", image)], boxes=[[api_box(cat)]])

    r = await upload(c, pid, [("a-again.png", image)], boxes=[[api_box(dog), api_box(dog, x=0.6, y=0.6, w=0.1, h=0.1)]])

    assert r.json()["results"][0]["value"]["isDuplicate"] is True
    (item,) = (await listing(c, pid))["items"]
    assert [a["classId"] for a in item["annotations"]] == [cat]  # not overwritten, not added to


async def test_malformed_boxes_reject_the_whole_upload_before_anything_is_stored(client, new_user, fake_s3):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    _, (foreign,) = await setup(c, "foreign")
    two = [("a.png", png(8, 8, 1)), ("b.png", png(8, 8, 2))]

    assert (await upload(c, pid, two, boxes=[[api_box(cat)]])).status_code == 422  # one entry for two files
    assert (await upload(c, pid, two, boxes=[None, [api_box(cat, x=0.9, w=0.5)]])).status_code == 422  # off the edge
    assert (await upload(c, pid, two, boxes=[None, [{"classId": cat, "x": 0.1}]])).status_code == 422  # missing keys
    assert (await upload(c, pid, two, boxes=[None, [api_box(foreign)]])).status_code == 400  # another project's class
    assert (await listing(c, pid))["total"] == 0 and fake_s3 == {}


async def test_a_class_id_or_boxes_are_refused_where_they_do_not_apply(client, new_user, fake_s3):
    c = await new_user()
    pid, (cat,) = await setup(c, "cat")
    one = [("a.png", png(8, 8))]
    r = await upload(c, pid, one, classId=cat)
    assert r.status_code == 400 and "send `boxes`" in r.json()["error"]["message"]  # a class is per box, not per image

    classification = await project(c, "image_classification")
    other_class = await make_class(c, classification["id"], "x")
    r = await upload(c, classification["id"], one, boxes=[[api_box(other_class)]])
    assert r.status_code == 400 and "object-detection" in r.json()["error"]["message"]
