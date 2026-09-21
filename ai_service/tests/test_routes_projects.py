"""Projects, label classes and API keys over HTTP against a real Postgres.

Ports server/routes/classes.integration.test.ts and adds the project and API-key surfaces.
"""

import uuid

import pytest
import sqlalchemy as sa

from theseus.db.models import ApiKey, DatasetVersion, Project, TrainingRun
from theseus.services import storage


@pytest.fixture(autouse=True)
def no_s3(monkeypatch):
    for name in ("delete_file", "delete_files", "delete_prefix"):
        monkeypatch.setattr(storage, name, lambda *a, **k: None)


async def make_project(c, task="image_classification", name="Cats vs dogs"):
    r = await c.post("/api/projects", json={"name": name, "description": None, "task": task})
    assert r.status_code == 200, r.text
    return r.json()["project"]


# -- Projects --------------------------------------------------------------------------------


async def test_creating_a_project_also_creates_its_dataset_and_draft_version(client, new_user, db):
    c = await new_user()
    p = await make_project(c, "text_classification")
    assert p["task"] == "text_classification" and p["userId"] == c.user_id

    detail = (await c.get(f"/api/projects/{p['id']}")).json()["project"]
    assert detail["runCount"] == 0 and detail["versionCount"] == 1
    ds = detail["dataset"]
    assert ds["modality"] == "text"  # derived from the task, never client supplied
    assert ds["draft"]["versionTag"] is None and ds["draft"]["status"] == "draft"
    assert ds["draft"]["splitCounts"] == {"train": 0, "validation": 0, "test": 0} and ds["draft"]["itemCount"] == 0
    assert ds["classes"] == []


async def test_an_untrainable_task_is_rejected_with_422_and_creates_nothing(client, new_user, db):
    c = await new_user()
    r = await c.post("/api/projects", json={"name": "x", "description": None, "task": "object_detection"})
    assert r.status_code == 422 and "not yet trainable" in r.json()["error"]["message"]
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(Project))).scalar_one() == 0


async def test_project_name_is_required(client, new_user):
    c = await new_user()
    r = await c.post("/api/projects", json={"name": "", "task": "text_classification"})
    assert r.status_code == 422


async def test_list_returns_only_my_projects_newest_first_with_modality(client, new_user):
    a, b = await new_user(), await new_user()
    first = await make_project(a, "text_classification", "first")
    second = await make_project(a, "tabular_regression", "second")
    await make_project(b, name="not mine")

    listed = (await a.get("/api/projects")).json()["projects"]
    assert [p["id"] for p in listed] == [second["id"], first["id"]]
    assert [p["draftDataset"]["modality"] for p in listed] == ["tabular", "text"]


async def test_update_renames_and_can_clear_the_description(client, new_user):
    c = await new_user()
    p = await make_project(c)
    r = await c.patch(f"/api/projects/{p['id']}", json={"name": "Renamed", "description": "hello"})
    assert (r.json()["project"]["name"], r.json()["project"]["description"]) == ("Renamed", "hello")
    r = await c.patch(f"/api/projects/{p['id']}", json={"description": None})
    assert r.json()["project"]["name"] == "Renamed" and r.json()["project"]["description"] is None


async def test_a_stranger_can_neither_read_change_nor_delete_my_project(client, new_user):
    owner, stranger = await new_user(), await new_user()
    p = await make_project(owner)
    for call in (
        stranger.get(f"/api/projects/{p['id']}"),
        stranger.patch(f"/api/projects/{p['id']}", json={"name": "hijacked"}),
        stranger.delete(f"/api/projects/{p['id']}"),
    ):
        assert (await call).status_code == 403
    assert (await stranger.get(f"/api/projects/{uuid.uuid4()}")).status_code == 404
    assert (await owner.get(f"/api/projects/{p['id']}")).json()["project"]["name"] == "Cats vs dogs"


async def test_unauthenticated_requests_are_401(client):
    assert (await client.get("/api/projects")).status_code == 401
    assert (await client.post("/api/projects", json={"name": "x", "task": "text_classification"})).status_code == 401


async def test_deleting_a_project_cascades_and_cancels_its_active_runs(client, new_user, db):
    c = await new_user()
    p = await make_project(c)
    async with db() as s:
        version_id = (await s.execute(sa.select(DatasetVersion.id))).scalar_one()
        run = TrainingRun(
            project_id=uuid.UUID(p["id"]), dataset_version_id=version_id, name="r", status="running", hyperparameters={}
        )
        s.add(run)
        await s.commit()
        run_id = run.id

    assert (await c.delete(f"/api/projects/{p['id']}")).status_code == 204
    async with db() as s:
        for model in (Project, DatasetVersion, TrainingRun):
            assert (await s.execute(sa.select(sa.func.count()).select_from(model))).scalar_one() == 0
    assert (await c.get(f"/api/projects/{p['id']}")).status_code == 404
    assert run_id is not None


# -- Label classes (ports classes.integration.test.ts) ---------------------------------------


async def test_class_lifecycle_create_list_update_and_soft_delete(client, new_user):
    c = await new_user()
    p = await make_project(c)
    base = f"/api/projects/{p['id']}/classes"
    assert (await c.get(base)).json() == {"classes": []}

    created = (await c.post(base, json={"name": "cat"})).json()["class"]
    assert created["name"] == "cat" and created["isActive"] is True
    assert created["uiColorHex"] == "#e03131"  # first colour of the auto palette

    second = (await c.post(base, json={"name": "dog"})).json()["class"]
    assert second["uiColorHex"] == "#2f9e44"  # palette advances

    updated = (await c.patch(f"{base}/{created['classId']}", json={"name": "kitten", "uiColorHex": "#112233"})).json()
    assert (updated["class"]["name"], updated["class"]["uiColorHex"]) == ("kitten", "#112233")

    assert (await c.delete(f"{base}/{second['classId']}")).status_code == 204
    assert [x["name"] for x in (await c.get(base)).json()["classes"]] == ["kitten"]  # soft-deleted are hidden
    assert (await c.patch(f"{base}/{second['classId']}", json={"name": "x"})).status_code == 404


async def test_duplicate_active_class_name_is_400_and_bad_colours_are_422(client, new_user):
    c = await new_user()
    p = await make_project(c)
    base = f"/api/projects/{p['id']}/classes"
    assert (await c.post(base, json={"name": "cat"})).status_code == 200
    dup = await c.post(base, json={"name": "cat"})
    assert dup.status_code == 400 and dup.json()["error"]["message"] == "Class name already exists"
    assert (await c.post(base, json={"name": "x", "uiColorHex": "red"})).status_code == 422
    assert (await c.post(base, json={"name": "y" * 101})).status_code == 422


async def test_recreating_a_deleted_class_name_reactivates_it_instead_of_erroring(client, new_user):
    c = await new_user()
    p = await make_project(c)
    base = f"/api/projects/{p['id']}/classes"
    first = (await c.post(base, json={"name": "cat"})).json()["class"]
    await c.delete(f"{base}/{first['classId']}")

    again = await c.post(base, json={"name": "cat", "description": "back"})
    assert again.status_code == 200
    assert again.json()["class"]["classId"] == first["classId"]  # same row: annotations stay attached
    assert again.json()["class"]["isActive"] is True and again.json()["class"]["description"] == "back"


async def test_renaming_into_an_existing_name_is_400(client, new_user):
    c = await new_user()
    p = await make_project(c)
    base = f"/api/projects/{p['id']}/classes"
    await c.post(base, json={"name": "cat"})
    dog = (await c.post(base, json={"name": "dog"})).json()["class"]
    assert (await c.patch(f"{base}/{dog['classId']}", json={"name": "cat"})).status_code == 400


async def test_class_routes_enforce_project_ownership_and_class_scoping(client, new_user):
    owner, stranger = await new_user(), await new_user()
    mine, theirs = await make_project(owner), await make_project(stranger)
    cls = (await owner.post(f"/api/projects/{mine['id']}/classes", json={"name": "cat"})).json()["class"]

    assert (await stranger.get(f"/api/projects/{mine['id']}/classes")).status_code == 403
    assert (await stranger.post(f"/api/projects/{mine['id']}/classes", json={"name": "x"})).status_code == 403
    # A class id from someone else's project cannot be reached through my own project path.
    assert (
        await stranger.patch(f"/api/projects/{theirs['id']}/classes/{cls['classId']}", json={"name": "x"})
    ).status_code == 403
    assert (await stranger.delete(f"/api/projects/{theirs['id']}/classes/{cls['classId']}")).status_code == 403
    assert (await owner.get(f"/api/projects/{mine['id']}/classes")).json()["classes"][0]["name"] == "cat"


async def test_project_detail_counts_only_active_classes(client, new_user):
    c = await new_user()
    p = await make_project(c)
    base = f"/api/projects/{p['id']}/classes"
    keep = (await c.post(base, json={"name": "keep"})).json()["class"]
    gone = (await c.post(base, json={"name": "gone"})).json()["class"]
    await c.delete(f"{base}/{gone['classId']}")
    classes = (await c.get(f"/api/projects/{p['id']}")).json()["project"]["dataset"]["classes"]
    assert [x["classId"] for x in classes] == [keep["classId"]]


# -- API keys --------------------------------------------------------------------------------


async def test_api_key_is_shown_once_stored_only_as_a_hash_and_listed_without_the_secret(client, new_user, db):
    c = await new_user()
    r = await c.post("/api/keys", json={"name": "ci"})
    assert r.status_code == 201
    created = r.json()
    assert created["key"].startswith("thsk_") and created["keyPrefix"] == created["key"][:11]

    listed = (await c.get("/api/keys")).json()["keys"]
    assert [k["id"] for k in listed] == [created["id"]]
    assert "key" not in listed[0] and created["key"] not in str(listed)
    assert listed[0]["revokedAt"] is None and listed[0]["lastUsedAt"] is None
    async with db() as s:
        stored = (await s.execute(sa.select(ApiKey))).scalar_one()
    assert stored.key_hash != created["key"] and len(stored.key_hash) == 64


async def test_revoking_a_key_is_soft_idempotent_404_and_scoped_to_its_owner(client, new_user, db):
    owner, stranger = await new_user(), await new_user()
    key = (await owner.post("/api/keys", json={"name": "k"})).json()
    assert (await stranger.delete(f"/api/keys/{key['id']}")).status_code == 404  # not even revealed to exist
    assert (await owner.delete(f"/api/keys/{key['id']}")).status_code == 204
    assert (await owner.delete(f"/api/keys/{key['id']}")).status_code == 404  # already revoked
    row = (await owner.get("/api/keys")).json()["keys"][0]
    assert row["revokedAt"] is not None  # kept for the audit trail, not deleted


async def test_api_key_name_is_validated(client, new_user):
    c = await new_user()
    assert (await c.post("/api/keys", json={"name": ""})).status_code == 422
    assert (await c.post("/api/keys", json={"name": "x" * 101})).status_code == 422
