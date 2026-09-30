"""/api/admin/runs, /exports and /system: the cross-user view of the job queues, and what an admin can do."""

import uuid

import pytest
import sqlalchemy as sa

from theseus.db.models import CustomModel, ModelExport, TrainingRun
from theseus.jobs.dispatcher import Dispatcher, Lane, set_dispatcher
from theseus.settings import get_settings

ADMIN_EMAIL = "boss@example.com"


@pytest.fixture
async def admin(monkeypatch, new_user):
    monkeypatch.setenv("THESEUS_ADMIN_EMAILS", ADMIN_EMAIL)
    get_settings.cache_clear()
    yield await new_user(ADMIN_EMAIL)
    get_settings.cache_clear()


async def test_only_admins_reach_these_routes(new_user):
    user = await new_user()
    for path in ("/api/admin/runs", "/api/admin/exports", "/api/admin/system"):
        assert (await user.get(path)).status_code == 403, path
    assert (await user.post(f"/api/admin/runs/{uuid.uuid4()}/cancel")).status_code == 403
    assert (await user.post(f"/api/admin/exports/{uuid.uuid4()}/requeue")).status_code == 403


# -- Runs --------------------------------------------------------------------------------------


async def test_runs_of_every_user_are_listed_with_their_owner_and_filterable(admin, make_run):
    queued, succeeded = await make_run(status="queued"), await make_run(status="succeeded")

    everything = (await admin.get("/api/admin/runs")).json()
    assert everything["total"] == 2 and {r["id"] for r in everything["runs"]} == {str(queued), str(succeeded)}
    row = next(r for r in everything["runs"] if r["id"] == str(queued))
    assert row["ownerEmail"].endswith("@x.co") and row["projectName"] == "p" and row["backend"] == "ludwig"
    assert row["task"] == "image_classification" and row["status"] == "queued"

    only = (await admin.get("/api/admin/runs", params={"status": "succeeded"})).json()
    assert [r["id"] for r in only["runs"]] == [str(succeeded)]
    assert (await admin.get("/api/admin/runs", params={"user": row["ownerEmail"][:8].upper()})).json()["total"] == 1
    assert (await admin.get("/api/admin/runs", params={"user": "%"})).json()["total"] == 0  # literal, not a wildcard
    assert (await admin.get("/api/admin/runs", params={"status": "nonsense"})).status_code == 422


async def test_run_list_is_paginated_newest_first(admin, make_run):
    ids = [await make_run() for _ in range(3)]
    page1 = (await admin.get("/api/admin/runs", params={"page_size": 2})).json()
    page2 = (await admin.get("/api/admin/runs", params={"page_size": 2, "page": 2})).json()
    assert page1["total"] == 3 and len(page1["runs"]) == 2 and len(page2["runs"]) == 1
    assert [r["id"] for r in page1["runs"] + page2["runs"]] == [str(i) for i in reversed(ids)]


async def test_an_admin_can_cancel_someone_elses_queued_run(admin, make_run, db):
    run_id = await make_run(status="queued")
    assert (await admin.post(f"/api/admin/runs/{run_id}/cancel")).status_code == 204
    async with db() as s:
        assert (await s.get(TrainingRun, run_id)).cancel_requested_at is not None


async def test_a_finished_or_unknown_run_cannot_be_cancelled(admin, make_run):
    done = await make_run(status="succeeded")
    r = await admin.post(f"/api/admin/runs/{done}/cancel")
    assert r.status_code == 409 and "already finished" in r.json()["error"]["message"]
    assert (await admin.post(f"/api/admin/runs/{uuid.uuid4()}/cancel")).status_code == 404


# -- Exports -----------------------------------------------------------------------------------


async def test_exports_of_every_user_are_listed_and_filterable(admin, make_export):
    failed, _ = await make_export(status="failed", failed_message="boom", last_error="boom", attempt=3)
    ready, _ = await make_export(status="ready")

    listing = (await admin.get("/api/admin/exports")).json()
    assert listing["total"] == 2
    row = next(e for e in listing["exports"] if e["id"] == str(failed))
    assert row["format"] == "onnx" and row["status"] == "failed" and row["attempt"] == 3 and row["lastError"] == "boom"
    assert row["ownerEmail"].endswith("@x.co") and row["runName"] == "run"

    only = (await admin.get("/api/admin/exports", params={"status": "ready"})).json()
    assert [e["id"] for e in only["exports"]] == [str(ready)]


async def test_a_failed_export_is_requeued_with_fresh_attempts(admin, make_export, db):
    export_id, _ = await make_export(
        status="failed", failed_message="boom", last_error="boom", attempt=3, claimed_by="old-worker"
    )
    assert (await admin.post(f"/api/admin/exports/{export_id}/requeue")).status_code == 204
    async with db() as s:
        e = await s.get(ModelExport, export_id)
    assert e.status == "pending" and e.attempt == 0 and e.failed_message is None and e.last_error is None
    assert e.claimed_by is None and e.lease_expires_at is None


async def test_only_a_failed_export_can_be_requeued(admin, make_export, db):
    for status in ("pending", "converting", "ready"):
        export_id, _ = await make_export(status=status)
        r = await admin.post(f"/api/admin/exports/{export_id}/requeue")
        assert r.status_code == 409 and "failed" in r.json()["error"]["message"], status
        async with db() as s:
            assert (await s.get(ModelExport, export_id)).status == status  # left alone
    assert (await admin.post(f"/api/admin/exports/{uuid.uuid4()}/requeue")).status_code == 404


# -- System ------------------------------------------------------------------------------------


async def test_system_overview_counts_and_reports_queue_depths(admin, new_user, make_run, make_export, db):
    await new_user()  # a second real user besides the admin
    await make_run(status="queued")
    await make_run(status="running")
    await make_run(status="succeeded")
    await make_export(status="pending")
    await make_export(status="ready")
    async with db() as s:
        s.add_all(
            [
                CustomModel(backend="ludwig", kind="hf_transformer", name="a", source_kind="hub", status="uploaded"),
                CustomModel(
                    backend="ludwig", kind="hf_transformer", name="b", source_kind="hub", status="ready", size_bytes=500
                ),
                CustomModel(
                    backend="ludwig",
                    kind="hf_transformer",
                    name="c",
                    source_kind="hub",
                    status="ready",
                    size_bytes=300,
                    archived_at=sa.func.now(),
                ),
            ]  # fmt: skip
        )
        await s.commit()

    body = (await admin.get("/api/admin/system")).json()
    counts = body["counts"]
    assert counts["admins"] == 1 and counts["disabledUsers"] == 0 and counts["users"] >= 2
    # Three runs made directly, plus the succeeded run each of the two exports had to be attached to.
    assert counts["runs"] == 5 and counts["activeRuns"] == 2 and counts["exports"] == 2
    assert counts["customModels"] == 2  # the archived one is not counted...
    assert counts["customModelBytes"] == 500 + 300  # ...but its files are still in storage, so its bytes are

    lanes = {lane["name"]: lane for lane in body["lanes"]}
    assert set(lanes) == {"train", "export", "validate"}
    assert lanes["train"]["queued"] == 1 and lanes["export"]["queued"] == 1 and lanes["validate"]["queued"] == 1
    assert lanes["train"]["running"] is None and lanes["train"]["capacity"] is None  # no dispatcher in this test

    backends = {b["id"]: b for b in body["backends"]}
    assert backends["ludwig"]["available"] is True and backends["ludwig"]["unavailableReason"] is None


async def test_system_overview_reports_lane_load_when_a_dispatcher_runs(admin):
    async def noop(_):  # never called: the dispatcher is not started
        return None

    from theseus.jobs import queue

    lane = Lane("train", queue.TRAIN, 1, noop)
    lane.running.add(uuid.uuid4())
    set_dispatcher(Dispatcher([lane, Lane("export", queue.EXPORT, 2, noop)]))
    try:
        lanes = {lane["name"]: lane for lane in (await admin.get("/api/admin/system")).json()["lanes"]}
    finally:
        set_dispatcher(None)
    assert (lanes["train"]["running"], lanes["train"]["capacity"]) == (1, 1)
    assert (lanes["export"]["running"], lanes["export"]["capacity"]) == (0, 2)
    assert lanes["validate"]["running"] is None  # a lane this dispatcher does not have
