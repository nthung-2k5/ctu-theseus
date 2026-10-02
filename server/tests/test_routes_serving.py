"""Export over HTTP: formats, requests, listing and download."""

import uuid

import pytest
import sqlalchemy as sa

from theseus.auth import rate_limit
from theseus.db.models import DatasetVersion, ModelExport, TrainingRun
from theseus.services import storage


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    store: dict[tuple[str, str], bytes] = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__((b, k), data))
    monkeypatch.setattr(storage, "delete_file", lambda b, k: store.pop((b, k), None))
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    rate_limit.reset()
    return store


async def succeeded_run(c, db, task="text_classification", status="succeeded"):
    p = (await c.post("/api/projects", json={"name": "p", "description": None, "task": task})).json()["project"]
    async with db() as s:
        vid = (
            await s.execute(sa.select(DatasetVersion.id).where(DatasetVersion.dataset_id == uuid.UUID(p["id"])))
        ).scalar_one()
        run = TrainingRun(
            project_id=uuid.UUID(p["id"]), dataset_version_id=vid, name="r", status=status, hyperparameters={}
        )
        s.add(run)
        await s.commit()
        return str(run.id)


# -- Export ----------------------------------------------------------------------------------


async def test_export_formats_are_listed_grouped_and_task_filtered(client, new_user, db):
    from theseus.export.formats.base import ExportFormat, _registry

    def modality_only(fmt_id, modality):
        class OnlyFor(ExportFormat):
            id = fmt_id
            label = fmt_id
            group = "Experimental"

            @classmethod
            def supports(cls, task):
                return task.modality == modality

            @classmethod
            def assemble(cls, ctx):
                pass

    modality_only("test_text_only", "text")
    modality_only("test_vision_only", "vision")
    try:
        c = await new_user()
        run = await succeeded_run(c, db)  # a text classification project
        everything = (await c.get("/api/export-formats")).json()["formats"]
        assert [f["id"] for f in everything][:2] == ["onnx", "torch_export"]
        assert {"id", "label", "description", "notice", "group", "artifact"} <= set(everything[0])
        assert {"test_text_only", "test_vision_only"} <= {f["id"] for f in everything}

        for_run = {f["id"] for f in (await c.get(f"/api/export-formats?runId={run}")).json()["formats"]}
        assert "test_text_only" in for_run and "test_vision_only" not in for_run

        stranger = await new_user()
        assert (await stranger.get(f"/api/export-formats?runId={run}")).status_code == 403
    finally:
        _registry.unregister("test_text_only")
        _registry.unregister("test_vision_only")


async def test_export_validation_rules(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    url = f"/api/runs/{run}/exports"
    msg = lambda r: r.json()["error"]["message"]  # noqa: E731
    assert (await c.post(url, json={"format": "torch_export"})).status_code == 202
    assert (await c.post(url, json={"format": "flutter_app"})).status_code == 202
    bad = await c.post(url, json={"format": "nope"})
    assert bad.status_code == 400 and "Unknown export format 'nope'" in msg(bad)
    assert (await c.post(url, json={})).status_code == 422
    unfinished = await succeeded_run(c, db, status="running")
    assert (await c.post(f"/api/runs/{unfinished}/exports", json={"format": "onnx"})).status_code == 409


async def test_an_export_that_does_not_support_the_projects_task_is_rejected(client, new_user, db):
    from theseus.export.formats.base import ExportFormat, _registry

    class NeverSupported(ExportFormat):
        id = "test_never_supported"
        label = "Never"

        @classmethod
        def supports(cls, task):
            return False

        @classmethod
        def assemble(cls, ctx):
            pass

    try:
        c = await new_user()
        run = await succeeded_run(c, db)
        r = await c.post(f"/api/runs/{run}/exports", json={"format": "test_never_supported"})
        assert r.status_code == 400 and "does not support this project's task" in r.json()["error"]["message"]
    finally:
        _registry.unregister("test_never_supported")


async def test_an_export_starts_pending_owned_by_the_project_owner_and_lists_newest_first(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    a = (await c.post(f"/api/runs/{run}/exports", json={"format": "onnx"})).json()["exportId"]
    b = (await c.post(f"/api/runs/{run}/exports", json={"format": "python_devkit"})).json()["exportId"]
    async with db() as s:
        rows = {str(e.id): e for e in (await s.execute(sa.select(ModelExport))).scalars()}
    assert rows[a].format == "onnx" and rows[b].format == "python_devkit"
    assert rows[a].status == "pending" and str(rows[a].user_id) == c.user_id
    assert rows[a].max_attempts == 3

    listed = (await c.get(f"/api/runs/{run}/exports")).json()["exports"]
    assert [e["id"] for e in listed] == [b, a]
    got = (await c.get(f"/api/exports/{a}")).json()["export"]
    assert got["status"] == "pending" and got["format"] == "onnx" and "tier" not in got and "lang" not in got


async def test_export_download_is_404_until_ready_then_redirects_and_is_owner_only(client, new_user, db):
    owner, stranger = await new_user(), await new_user()
    run = await succeeded_run(owner, db)
    eid = (await owner.post(f"/api/runs/{run}/exports", json={"format": "onnx"})).json()["exportId"]
    assert (await owner.get(f"/api/exports/{eid}/download")).status_code == 404
    async with db() as s:
        await s.execute(
            sa.update(ModelExport)
            .where(ModelExport.id == uuid.UUID(eid))
            .values(status="ready", bundle_key=f"{run}/bundles/{eid}.zip")
        )
        await s.commit()
    r = await owner.get(f"/api/exports/{eid}/download", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == f"https://s3.test/theseus-models/{run}/bundles/{eid}.zip"
    for url in (f"/api/exports/{eid}", f"/api/exports/{eid}/download", f"/api/runs/{run}/exports"):
        assert (await stranger.get(url)).status_code == 403
    assert (await stranger.post(f"/api/runs/{run}/exports", json={"format": "onnx"})).status_code == 403
