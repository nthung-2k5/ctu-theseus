"""/api/admin/workers: registering, switching off and revoking remote training workers, and run views."""

import uuid
from datetime import timedelta

import pytest
import sqlalchemy as sa

from omnitrain.db.models import TrainingRun, TrainingWorker
from omnitrain.events import get_event_writer
from omnitrain.schemas.worker import PROTOCOL_VERSION
from omnitrain.settings import get_settings

ADMIN_EMAIL = "boss@example.com"
CLAIM = {"backends": ["pytorch"], "protocol": PROTOCOL_VERSION, "version": "1.2.3", "hostname": "box", "gpu": "RTX"}


@pytest.fixture
async def admin(monkeypatch, new_user):
    monkeypatch.setenv("OMNITRAIN_ADMIN_EMAILS", ADMIN_EMAIL)
    get_settings.cache_clear()
    yield await new_user(ADMIN_EMAIL)
    get_settings.cache_clear()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


async def register(admin, name="gpu-box"):
    r = await admin.post("/api/admin/workers", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


async def test_only_admins_reach_these_routes(new_user):
    user = await new_user()
    assert (await user.get("/api/admin/workers")).status_code == 403
    assert (await user.post("/api/admin/workers", json={"name": "x"})).status_code == 403
    assert (await user.patch(f"/api/admin/workers/{uuid.uuid4()}", json={"enabled": False})).status_code == 403
    assert (await user.delete(f"/api/admin/workers/{uuid.uuid4()}")).status_code == 403


async def test_registering_a_worker_returns_its_token_once_and_it_works(admin, client, db):
    created = await register(admin, "  lab rtx  ")
    token = created["token"]
    assert token.startswith("thwk_") and created["worker"]["name"] == "lab rtx"
    assert created["worker"]["online"] is False and created["worker"]["backends"] == []

    async with db() as s:  # only a hash is stored
        stored = (await s.execute(sa.select(TrainingWorker.token_hash))).scalar_one()
    assert stored != token and token not in stored
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=bearer(token))).status_code == 204

    listed = (await admin.get("/api/admin/workers")).json()["workers"]
    assert [w["name"] for w in listed] == ["lab rtx"] and "token" not in listed[0]


async def test_a_worker_shows_what_it_reported_and_is_online_only_while_recent(admin, client, db):
    created = await register(admin)
    await client.post("/api/worker/claim", json=CLAIM, headers=bearer(created["token"]))

    worker = (await admin.get("/api/admin/workers")).json()["workers"][0]
    assert worker["online"] is True
    assert (worker["hostname"], worker["gpu"], worker["version"], worker["backends"]) == (
        "box",
        "RTX",
        "1.2.3",
        ["pytorch"],
    )

    async with db() as s:
        await s.execute(sa.update(TrainingWorker).values(last_seen_at=sa.func.now() - timedelta(minutes=5)))
        await s.commit()
    assert (await admin.get("/api/admin/workers")).json()["workers"][0]["online"] is False


async def test_the_list_says_which_run_each_worker_is_training(admin, client, make_run):
    created = await register(admin)
    rid = await make_run()
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=bearer(created["token"]))).status_code == 200

    worker = (await admin.get("/api/admin/workers")).json()["workers"][0]
    assert (worker["currentRunId"], worker["currentRunName"]) == (str(rid), "run")


async def test_switching_a_worker_off_stops_new_claims_but_not_the_run_it_has(admin, client, db, make_run):
    created = await register(admin)
    token, worker_id = created["token"], created["worker"]["id"]
    running = await make_run()
    await client.post("/api/worker/claim", json=CLAIM, headers=bearer(token))
    queued = await make_run()

    r = await admin.patch(f"/api/admin/workers/{worker_id}", json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False

    assert (await client.post("/api/worker/claim", json=CLAIM, headers=bearer(token))).status_code == 204
    events = await client.post(f"/api/worker/runs/{running}/events", json={}, headers=bearer(token))
    assert events.status_code == 200  # still authenticated: it may finish its run
    async with db() as s:
        assert (await s.execute(sa.select(TrainingRun.status).where(TrainingRun.id == queued))).scalar_one() == "queued"


async def test_revoking_a_worker_ends_its_token_and_fails_the_run_it_was_training(admin, client, db, make_run):
    created = await register(admin)
    token, worker_id = created["token"], created["worker"]["id"]
    rid = await make_run()
    await client.post("/api/worker/claim", json=CLAIM, headers=bearer(token))

    assert (await admin.delete(f"/api/admin/workers/{worker_id}")).status_code == 204
    await get_event_writer().flush()

    assert (await client.post("/api/worker/claim", json=CLAIM, headers=bearer(token))).status_code == 401
    async with db() as s:
        run = (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == rid))).scalar_one()
    assert run.status == "failed" and "revoked" in run.failed_message
    listed = (await admin.get("/api/admin/workers")).json()["workers"][0]
    assert listed["revokedAt"] is not None and listed["online"] is False and listed["enabled"] is False
    assert (await admin.delete(f"/api/admin/workers/{worker_id}")).status_code == 404  # already revoked
    assert (await admin.patch(f"/api/admin/workers/{worker_id}", json={"enabled": True})).status_code == 404


async def test_an_unknown_worker_is_a_404(admin):
    missing = uuid.uuid4()
    assert (await admin.patch(f"/api/admin/workers/{missing}", json={"enabled": True})).status_code == 404
    assert (await admin.delete(f"/api/admin/workers/{missing}")).status_code == 404


async def test_a_run_says_where_it_trained_to_its_owner_and_to_the_admin(admin, db, make_run, make_worker):
    worker = await make_worker(token="thwk_x")
    remote = await make_run(status="running", worker_id=worker)
    local = await make_run(status="running")

    rows = {r["id"]: r for r in (await admin.get("/api/admin/runs")).json()["runs"]}
    assert rows[str(remote)]["workerName"] == "gpu-box" and rows[str(local)]["workerName"] is None

    async with db() as s:  # make the admin the owner of both runs' projects, to read the owner's view
        await s.execute(sa.text("update projects set user_id = :u"), {"u": uuid.UUID(admin.user_id)})
        await s.commit()
    assert (await admin.get(f"/api/runs/{remote}")).json()["run"]["workerName"] == "gpu-box"
    assert (await admin.get(f"/api/runs/{local}")).json()["run"]["workerName"] is None


async def test_the_system_overview_counts_workers_and_those_online(admin, client, make_worker):
    await make_worker(token="thwk_a")
    await make_worker(token="thwk_b", last_seen_at=None)
    await make_worker(token="thwk_c", revoked_at=sa.func.now())

    counts = (await admin.get("/api/admin/system")).json()["counts"]
    assert (counts["workers"], counts["workersOnline"]) == (2, 1)
