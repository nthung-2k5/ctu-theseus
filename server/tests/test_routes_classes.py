"""Label classes over HTTP: creation-order listing and the batch save."""

import uuid


async def project(c, task="text_classification"):
    return (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]


async def make_class(c, pid, name, **extra):
    r = await c.post(f"/api/projects/{pid}/classes", json={"name": name, **extra})
    assert r.status_code == 200, r.text
    return r.json()["class"]


async def listing(c, pid):
    r = await c.get(f"/api/projects/{pid}/classes")
    assert r.status_code == 200, r.text
    return r.json()["classes"]


def names(classes):
    return [c["name"] for c in classes]


async def save(c, pid, entries):
    return await c.put(f"/api/projects/{pid}/classes", json={"classes": entries})


# -- Listing -----------------------------------------------------------------------------------


async def test_classes_are_listed_in_creation_order(client, new_user):
    c = await new_user()
    p = await project(c)
    for name in ("b", "a", "c"):
        await make_class(c, p["id"], name)
    assert names(await listing(c, p["id"])) == ["b", "a", "c"]
    detail = (await c.get(f"/api/projects/{p['id']}")).json()["project"]
    assert names(detail["dataset"]["classes"]) == ["b", "a", "c"]


# -- Batch save --------------------------------------------------------------------------------


async def test_save_renames_recolors_describes_adds_and_deletes_in_one_call(client, new_user):
    c = await new_user()
    p = await project(c)
    a = await make_class(c, p["id"], "a", uiColorHex="#111111")
    b = await make_class(c, p["id"], "b", uiColorHex="#222222", description="old")
    dropped = await make_class(c, p["id"], "drop me")

    r = await save(
        c,
        p["id"],
        [
            {"name": "brand new", "uiColorHex": "#333333"},
            {"classId": b["classId"], "name": "B renamed", "description": None, "uiColorHex": "#444444"},
            {"classId": a["classId"], "name": "a"},
        ],
    )
    assert r.status_code == 200, r.text
    got = r.json()["classes"]
    assert names(got) == ["a", "B renamed", "brand new"]  # existing classes keep their place, new ones go last
    by_name = {x["name"]: x for x in got}
    assert by_name["B renamed"]["classId"] == b["classId"]  # renamed in place, same id
    assert by_name["B renamed"]["description"] is None and by_name["B renamed"]["uiColorHex"] == "#444444"
    assert by_name["a"]["uiColorHex"] == "#111111"  # colour omitted: kept
    assert by_name["brand new"]["uiColorHex"] == "#333333"
    assert dropped["classId"] not in {x["classId"] for x in got}
    assert names(await listing(c, p["id"])) == ["a", "B renamed", "brand new"]


async def test_save_new_class_without_a_colour_gets_a_palette_colour(client, new_user):
    c = await new_user()
    p = await project(c)
    r = await save(c, p["id"], [{"name": "x"}, {"name": "y"}])
    colors = [x["uiColorHex"] for x in r.json()["classes"]]
    assert all(colors) and colors[0] != colors[1]


async def test_save_can_swap_two_names(client, new_user):
    c = await new_user()
    p = await project(c)
    a = await make_class(c, p["id"], "a")
    b = await make_class(c, p["id"], "b")
    r = await save(c, p["id"], [{"classId": a["classId"], "name": "b"}, {"classId": b["classId"], "name": "a"}])
    assert r.status_code == 200, r.text
    got = {x["classId"]: x["name"] for x in r.json()["classes"]}
    assert got == {a["classId"]: "b", b["classId"]: "a"}


async def test_save_can_reuse_the_name_of_a_class_deleted_earlier(client, new_user):
    c = await new_user()
    p = await project(c)
    old = await make_class(c, p["id"], "cat")
    keep = await make_class(c, p["id"], "dog")
    await c.delete(f"/api/projects/{p['id']}/classes/{old['classId']}")
    # Renaming a live class onto the tombstone's name must not trip the unique constraint.
    r = await save(c, p["id"], [{"classId": keep["classId"], "name": "cat"}])
    assert r.status_code == 200, r.text
    assert [(x["classId"], x["name"]) for x in r.json()["classes"]] == [(keep["classId"], "cat")]


async def test_save_renaming_onto_a_class_it_deletes_in_the_same_call_works(client, new_user):
    c = await new_user()
    p = await project(c)
    cat = await make_class(c, p["id"], "cat")
    dog = await make_class(c, p["id"], "dog")
    r = await save(c, p["id"], [{"classId": dog["classId"], "name": "cat"}])
    assert r.status_code == 200, r.text
    assert [(x["classId"], x["name"]) for x in r.json()["classes"]] == [(dog["classId"], "cat")]
    assert cat["classId"] not in {x["classId"] for x in r.json()["classes"]}


async def test_save_revives_a_deleted_class_when_an_entry_reuses_its_name(client, new_user):
    c = await new_user()
    p = await project(c)
    old = await make_class(c, p["id"], "cat")
    await c.delete(f"/api/projects/{p['id']}/classes/{old['classId']}")
    r = await save(c, p["id"], [{"name": "cat"}])
    assert r.status_code == 200, r.text
    assert [(x["classId"], x["name"]) for x in r.json()["classes"]] == [(old["classId"], "cat")]


async def test_save_with_an_empty_list_deletes_every_class(client, new_user):
    c = await new_user()
    p = await project(c)
    await make_class(c, p["id"], "a")
    r = await save(c, p["id"], [])
    assert r.status_code == 200 and r.json()["classes"] == []
    assert await listing(c, p["id"]) == []


async def test_save_normalizes_whitespace_and_rejects_duplicate_names_ignoring_case(client, new_user):
    c = await new_user()
    p = await project(c)
    ok = await save(c, p["id"], [{"name": "  big   cat  "}])
    assert names(ok.json()["classes"]) == ["big cat"]
    dup = await save(c, p["id"], [{"name": "Cat"}, {"name": "cat"}])
    assert dup.status_code == 400 and "Duplicate" in dup.json()["error"]["message"]
    assert names(await listing(c, p["id"])) == ["big cat"]  # the failed save changed nothing


async def test_save_rejects_blank_bad_colour_and_oversized_lists(client, new_user):
    c = await new_user()
    p = await project(c)
    assert (await save(c, p["id"], [{"name": "   "}])).status_code == 422
    assert (await save(c, p["id"], [{"name": "a", "uiColorHex": "red"}])).status_code == 422
    assert (await save(c, p["id"], [{"name": f"c{i}"} for i in range(201)])).status_code == 422


async def test_save_rejects_unknown_deleted_or_repeated_class_ids(client, new_user):
    c = await new_user()
    p = await project(c)
    a = await make_class(c, p["id"], "a")
    gone = await make_class(c, p["id"], "gone")
    await c.delete(f"/api/projects/{p['id']}/classes/{gone['classId']}")

    unknown = await save(c, p["id"], [{"classId": str(uuid.uuid4()), "name": "x"}])
    assert unknown.status_code == 404
    deleted = await save(c, p["id"], [{"classId": gone["classId"], "name": "gone"}])
    assert deleted.status_code == 404
    twice = await save(c, p["id"], [{"classId": a["classId"], "name": "a"}, {"classId": a["classId"], "name": "again"}])
    assert twice.status_code == 400
    assert names(await listing(c, p["id"])) == ["a"]


async def test_save_cannot_touch_another_users_classes(client, new_user):
    owner, other = await new_user(), await new_user()
    p = await project(owner)
    cls = await make_class(owner, p["id"], "mine")
    r = await save(other, p["id"], [])
    assert r.status_code in (403, 404)
    # A class id from someone else's project is not addressable through your own project.
    mine = await project(other)
    r = await save(other, mine["id"], [{"classId": cls["classId"], "name": "stolen"}])
    assert r.status_code == 404
    assert names(await listing(owner, p["id"])) == ["mine"]


async def test_save_keeps_annotations_attached_to_a_renamed_class(client, new_user):
    c = await new_user()
    p = await project(c)
    cls = await make_class(c, p["id"], "cat")
    item = {"split": "train", "textFeatures": {"rawText": "meow"}}
    item["annotations"] = [{"annotationType": "classification", "classId": cls["classId"]}]
    assert (await c.post(f"/api/projects/{p['id']}/items", json={"items": [item]})).status_code == 200
    r = await save(c, p["id"], [{"classId": cls["classId"], "name": "feline"}])
    assert r.status_code == 200, r.text
    page = (await c.get(f"/api/projects/{p['id']}/items")).json()
    assert [(x["classId"], x["count"]) for x in page["classCounts"]] == [(cls["classId"], 1)]


# -- Stats -------------------------------------------------------------------------------------


async def add_items(c, pid, *specs):
    """Add inline text items; each spec is (split, class_id or None)."""
    items = []
    for i, (split, cls) in enumerate(specs):
        item = {"split": split, "textFeatures": {"rawText": f"row {i}"}}
        if cls:
            item["annotations"] = [{"annotationType": "classification", "classId": cls}]
        items.append(item)
    r = await c.post(f"/api/projects/{pid}/items", json={"items": items})
    assert r.status_code == 200, r.text
    return [x["id"] for x in r.json()["created"]]


async def stats(c, pid):
    r = await c.get(f"/api/projects/{pid}/classes/stats")
    assert r.status_code == 200, r.text
    return r.json()


async def test_stats_cover_every_active_class_including_empty_ones_with_per_split_counts(client, new_user):
    c = await new_user()
    p = await project(c)
    cat, dog, empty = [(await make_class(c, p["id"], n))["classId"] for n in ("cat", "dog", "empty")]
    await add_items(
        c, p["id"], ("train", cat), ("train", cat), ("validation", cat), ("test", dog), ("train", None), ("test", None)
    )
    body = await stats(c, p["id"])
    by_id = {s["classId"]: s for s in body["stats"]}
    assert [s["classId"] for s in body["stats"]] == [cat, dog, empty]  # creation order, the empty class included
    assert by_id[cat] == {"classId": cat, "total": 3, "train": 2, "validation": 1, "test": 0}
    assert by_id[dog] == {"classId": dog, "total": 1, "train": 0, "validation": 0, "test": 1}
    assert by_id[empty] == {"classId": empty, "total": 0, "train": 0, "validation": 0, "test": 0}
    assert (body["total"], body["unassigned"], body["retired"]) == (6, 2, 0)
    assert body["minItemsPerClass"] == 3


async def test_items_of_a_removed_class_count_as_retired_not_as_a_class(client, new_user):
    c = await new_user()
    p = await project(c)
    keep, gone = (await make_class(c, p["id"], "keep"))["classId"], (await make_class(c, p["id"], "gone"))["classId"]
    await add_items(c, p["id"], ("train", keep), ("train", gone), ("train", gone))
    assert (await save(c, p["id"], [{"classId": keep, "name": "keep"}])).status_code == 200
    body = await stats(c, p["id"])
    assert [s["classId"] for s in body["stats"]] == [keep]
    assert (body["total"], body["retired"], body["unassigned"]) == (3, 2, 0)


async def test_stats_are_for_class_tasks_only_and_owner_only(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    assert (await stranger.get(f"/api/projects/{p['id']}/classes/stats")).status_code in (403, 404)
    regression = await project(owner, "tabular_regression")
    assert (await owner.get(f"/api/projects/{regression['id']}/classes/stats")).status_code == 400


# -- Merging on save ---------------------------------------------------------------------------


async def test_a_merge_moves_the_removed_classes_items_into_the_kept_class(client, new_user):
    c = await new_user()
    p = await project(c)
    cat, kitten, dog = [await make_class(c, p["id"], n) for n in ("cat", "kitten", "dog")]
    await add_items(
        c, p["id"], ("train", cat["classId"]), ("train", kitten["classId"]), ("validation", kitten["classId"])
    )
    r = await c.put(
        f"/api/projects/{p['id']}/classes",
        json={
            "classes": [{"classId": cat["classId"], "name": "cat"}, {"classId": dog["classId"], "name": "dog"}],
            "merges": [{"fromClassId": kitten["classId"], "intoClassId": cat["classId"]}],
        },
    )
    assert r.status_code == 200, r.text
    assert names(r.json()["classes"]) == ["cat", "dog"]
    body = await stats(c, p["id"])
    by_id = {s["classId"]: s for s in body["stats"]}
    assert by_id[cat["classId"]]["total"] == 3 and by_id[cat["classId"]]["validation"] == 1
    assert by_id[dog["classId"]]["total"] == 0
    assert (body["unassigned"], body["retired"]) == (0, 0)  # nothing was left unlabelled or orphaned


async def test_a_merge_target_may_be_renamed_in_the_same_save(client, new_user):
    c = await new_user()
    p = await project(c)
    a, b = [await make_class(c, p["id"], n) for n in ("a", "b")]
    await add_items(c, p["id"], ("train", a["classId"]), ("train", b["classId"]))
    r = await c.put(
        f"/api/projects/{p['id']}/classes",
        json={
            "classes": [{"classId": b["classId"], "name": "b renamed"}],
            "merges": [{"fromClassId": a["classId"], "intoClassId": b["classId"]}],
        },
    )
    assert r.status_code == 200, r.text
    assert (await stats(c, p["id"]))["stats"][0]["total"] == 2


async def test_merge_validation(client, new_user):
    c = await new_user()
    p = await project(c)
    other = await project(c)
    a, b, c3 = [await make_class(c, p["id"], n) for n in ("a", "b", "c")]
    foreign = await make_class(c, other["id"], "foreign")
    keep_ab = [{"classId": a["classId"], "name": "a"}, {"classId": b["classId"], "name": "b"}]

    async def put(entries, merges):
        return await c.put(f"/api/projects/{p['id']}/classes", json={"classes": entries, "merges": merges})

    def merge(src, dst):
        return {"fromClassId": src["classId"], "intoClassId": dst["classId"]}

    # The source must be a class this save removes.
    assert (await put(keep_ab, [merge(a, b)])).status_code == 400
    # The target must be a class this save keeps.
    assert (await put(keep_ab, [merge(c3, c3)])).status_code == 400
    removed_target = [{"classId": a["classId"], "name": "a"}]
    assert (await put(removed_target, [merge(c3, b)])).status_code == 400
    # A source is merged once.
    assert (await put(removed_target, [merge(b, a), merge(b, a)])).status_code == 400
    # Another project's classes are neither a source nor a target.
    assert (await put(removed_target, [merge(foreign, a)])).status_code == 404
    assert (await put(removed_target, [merge(b, foreign)])).status_code == 400
    # None of the rejected saves changed anything.
    assert names(await listing(c, p["id"])) == ["a", "b", "c"]


async def test_a_save_without_merges_still_leaves_removed_items_unlabelled(client, new_user):
    c = await new_user()
    p = await project(c)
    a, b = [await make_class(c, p["id"], n) for n in ("a", "b")]
    await add_items(c, p["id"], ("train", b["classId"]))
    assert (await save(c, p["id"], [{"classId": a["classId"], "name": "a"}])).status_code == 200
    body = await stats(c, p["id"])
    assert body["retired"] == 1 and body["stats"][0]["total"] == 0
