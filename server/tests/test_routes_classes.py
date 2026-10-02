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
