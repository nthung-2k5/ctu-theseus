"""/api/worker: the routes a remote training worker calls. Auth, claiming, progress, and the final report."""

import pytest
import sqlalchemy as sa

from omnitrain.db.models import RunEvaluation, TrainingMetric, TrainingRun, TrainingWorker
from omnitrain.events import get_event_writer
from omnitrain.schemas.worker import PROTOCOL_VERSION
from omnitrain.services import storage

TOKEN = "thwk_test-token"
OTHER_TOKEN = "thwk_other-token"
CLAIM = {"backends": ["ludwig"], "protocol": PROTOCOL_VERSION, "version": "1.2.3", "hostname": "box", "gpu": "RTX"}


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


async def run_row(db, run_id):
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()


async def worker_row(db, worker_id):
    async with db() as s:
        return (await s.execute(sa.select(TrainingWorker).where(TrainingWorker.id == worker_id))).scalar_one()


async def claimed_run(client, db, make_run, make_worker, **run_fields):
    """A worker (token TOKEN) that has claimed a run: returns (worker_id, run_id)."""
    worker = await make_worker(token=TOKEN)
    rid = await make_run(**run_fields)
    r = await client.post("/api/worker/claim", json=CLAIM, headers=auth())
    assert r.status_code == 200 and r.json()["id"] == str(rid), r.text
    return worker, rid


# -- Authentication ----------------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer thwk_nope"}, {"Authorization": "thwk_test-token"}])
async def test_a_request_without_a_valid_worker_token_is_refused(client, make_worker, headers):
    await make_worker(token=TOKEN)
    r = await client.post("/api/worker/claim", json=CLAIM, headers=headers)
    assert r.status_code == 401


async def test_a_revoked_worker_is_refused_and_a_user_session_is_not_a_worker(client, db, make_worker, new_user):
    worker = await make_worker(token=TOKEN, revoked_at=sa.func.now(), last_seen_at=None)
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 401
    user = await new_user()  # a logged-in browser session carries no worker token
    assert (await user.post("/api/worker/claim", json=CLAIM)).status_code == 401
    assert (await worker_row(db, worker)).last_seen_at is None  # a revoked worker is not "seen"


async def test_every_call_marks_the_worker_as_seen(client, db, make_worker):
    worker = await make_worker(token=TOKEN, last_seen_at=None)
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 204
    assert (await worker_row(db, worker)).last_seen_at is not None


async def test_worker_routes_are_exempt_from_the_cross_origin_check_because_they_use_no_cookies(client, make_worker):
    await make_worker(token=TOKEN)
    r = await client.post("/api/worker/claim", json=CLAIM, headers={**auth(), "Origin": "https://evil.example"})
    assert r.status_code == 204


# -- Claiming ----------------------------------------------------------------------------------


async def test_a_claim_hands_out_the_run_and_starts_it(client, db, app, make_run, make_worker):
    worker, rid = await claimed_run(client, db, make_run, make_worker, backend="ludwig")
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert (run.status, run.worker_id, run.started_at is not None) == ("running", worker, True)
    info = (await worker_row(db, worker)).info
    assert info["backends"] == ["ludwig"] and info["gpu"] == "RTX" and info["hostname"] == "box"


async def test_the_spec_carries_what_run_core_needs(client, db, make_run, make_worker):
    await make_worker(token=TOKEN)
    rid = await make_run(config={"a": 1})
    spec = (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).json()
    run = await run_row(db, rid)
    assert spec == {
        "id": str(rid),
        "backend": "ludwig",
        "config": {"a": 1},
        "datasetVersionId": str(run.dataset_version_id),
        "configKey": "cfg",
        "customModel": None,
    }


async def test_a_custom_model_run_comes_with_what_the_worker_needs_to_fetch_the_model(
    client, db, make_run, make_worker
):
    from omnitrain.db.models import CustomModel
    from omnitrain.services import custom_models

    await make_worker(("video",), token=TOKEN)
    async with db() as s:
        model = CustomModel(
            backend="video", kind="hf_video", name="VideoMAE", source_kind="upload", sha256="a" * 64, spec={"x": 1}
        )
        s.add(model)
        await s.commit()
        server_path = str(custom_models.local_dir_for(model))
        model_id = model.id
    rid = await make_run(backend="video", custom_model_id=model_id)

    r = await client.post("/api/worker/claim", json={**CLAIM, "backends": ["video"]}, headers=auth())

    assert r.status_code == 200 and r.json()["id"] == str(rid)
    assert r.json()["customModel"] == {
        "id": str(model_id),
        "kind": "hf_video",
        "sourceKind": "upload",
        "sourceRef": None,
        "revision": None,
        "sha256": "a" * 64,
        "spec": {"x": 1},
        "serverPath": server_path,
    }


async def test_nothing_queued_or_a_backend_it_lacks_means_204(client, make_run, make_worker):
    await make_worker(token=TOKEN)
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 204
    await make_run(backend="yolo")
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 204


async def test_a_disabled_worker_claims_nothing_but_is_still_heard(client, db, make_run, make_worker):
    worker = await make_worker(token=TOKEN, enabled=False)
    rid = await make_run()
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 204
    assert (await run_row(db, rid)).status == "queued"
    assert (await worker_row(db, worker)).info["backends"] == ["ludwig"]


async def test_a_worker_on_a_different_protocol_is_told_so_and_advertises_no_backends(
    client, db, make_run, make_worker
):
    worker = await make_worker(token=TOKEN)
    rid = await make_run()
    r = await client.post("/api/worker/claim", json={**CLAIM, "protocol": PROTOCOL_VERSION + 1}, headers=auth())
    assert r.status_code == 409 and f"protocol {PROTOCOL_VERSION + 1}" in r.json()["error"]["message"]
    assert (await run_row(db, rid)).status == "queued"
    # so the server does not hold runs back waiting for a worker that cannot take them
    assert (await worker_row(db, worker)).info["backends"] == []


async def test_a_run_canceled_while_queued_is_ended_not_handed_out(client, db, make_run, make_worker):
    await make_worker(token=TOKEN)
    rid = await make_run(cancel_requested_at=sa.func.now())
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).status_code == 204
    await get_event_writer().flush()
    assert (await run_row(db, rid)).status == "canceled"


# -- Progress ----------------------------------------------------------------------------------


async def test_events_become_metrics_and_best_epoch(client, db, make_run, make_worker):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    body = {
        "metrics": [
            {"epoch": 1, "split": "train", "metrics": {"loss": 1.0}},
            {"epoch": 1, "split": "validation", "metrics": {"loss": 0.9}},
            {"epoch": 2, "split": "validation", "metrics": {"loss": 0.4}},
        ],
        "heartbeat": True,
    }
    r = await client.post(f"/api/worker/runs/{rid}/events", json=body, headers=auth())
    assert r.status_code == 200 and r.json() == {"cancel": False, "terminal": False}
    await get_event_writer().flush()

    async with db() as s:
        n = (await s.execute(sa.select(sa.func.count()).select_from(TrainingMetric))).scalar_one()
    assert n == 3
    assert (await run_row(db, rid)).best_epoch == 2


async def test_resending_a_metric_changes_nothing(client, db, make_run, make_worker):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    body = {"metrics": [{"epoch": 1, "split": "train", "metrics": {"loss": 1.0}}]}
    for _ in range(2):
        assert (await client.post(f"/api/worker/runs/{rid}/events", json=body, headers=auth())).status_code == 200
        await get_event_writer().flush()
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingMetric))).scalar_one() == 1


async def test_a_cancel_reaches_the_worker_through_its_next_events_call(client, db, make_run, make_worker):
    from omnitrain.jobs import abort

    _, rid = await claimed_run(client, db, make_run, make_worker)
    assert await abort.request_cancel(rid) is True
    await get_event_writer().flush()

    r = await client.post(f"/api/worker/runs/{rid}/events", json={}, headers=auth())
    assert r.json() == {"cancel": True, "terminal": True}  # canceled is also already over on the server


async def test_events_for_a_run_that_has_ended_are_dropped_and_say_so(client, db, make_run, make_worker):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    async with db() as s:
        await s.execute(sa.update(TrainingRun).where(TrainingRun.id == rid).values(status="failed"))
        await s.commit()
    body = {"metrics": [{"epoch": 1, "split": "train", "metrics": {"loss": 1.0}}]}
    r = await client.post(f"/api/worker/runs/{rid}/events", json=body, headers=auth())
    assert r.json()["terminal"] is True
    await get_event_writer().flush()
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingMetric))).scalar_one() == 0


async def test_a_worker_cannot_touch_a_run_it_did_not_claim(client, db, make_run, make_worker):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    await make_worker(token=OTHER_TOKEN)
    for path, body in (("events", {}), ("complete", {"status": "failed"})):
        r = await client.post(f"/api/worker/runs/{rid}/{path}", json=body, headers=auth(OTHER_TOKEN))
        assert r.status_code == 404
    queued = await make_run()  # never claimed by anyone
    assert (await client.post(f"/api/worker/runs/{queued}/events", json={}, headers=auth())).status_code == 404
    assert (await run_row(db, rid)).status == "running"


# -- Completing --------------------------------------------------------------------------------


@pytest.fixture
def results_in_storage(monkeypatch):
    present = {"keys": ["rid/results/model.pt"]}
    monkeypatch.setattr(storage, "list_keys", lambda bucket, prefix: present["keys"])
    return present


EVALUATION = {
    "status": "success",
    "split": "test",
    "report": {"split": "test", "overall": {"accuracy": 0.9, "macroF1": 0.85}},
    "accuracy": 0.9,
    "macroF1": 0.85,
}


async def test_success_records_the_evaluation_under_server_derived_keys(
    client, db, make_run, make_worker, results_in_storage
):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    hostile = {**EVALUATION, "reportKey": "someone-elses/report.json", "predictionsKey": "../../etc/passwd"}

    r = await client.post(
        f"/api/worker/runs/{rid}/complete", json={"status": "succeeded", "evaluation": hostile}, headers=auth()
    )
    assert r.status_code == 200 and r.json() == {"status": "succeeded"}
    await get_event_writer().flush()

    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        ev = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert (ev.status, ev.split) == ("success", "test")
    assert ev.accuracy == pytest.approx(0.9) and ev.macro_f1 == pytest.approx(0.85)
    assert ev.report_key == f"{rid}/evaluation/report.json"
    assert ev.predictions_key == f"{rid}/evaluation/predictions.parquet"


async def test_success_without_an_evaluation_is_fine(client, db, make_run, make_worker, results_in_storage):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    r = await client.post(f"/api/worker/runs/{rid}/complete", json={"status": "succeeded"}, headers=auth())
    assert r.status_code == 200
    await get_event_writer().flush()
    assert (await run_row(db, rid)).status == "succeeded"


async def test_a_failed_evaluation_is_recorded_but_the_run_still_succeeds(
    client, db, make_run, make_worker, results_in_storage
):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    body = {"status": "succeeded", "evaluation": {"status": "failed", "failedMessage": "matrix bug"}}
    assert (await client.post(f"/api/worker/runs/{rid}/complete", json=body, headers=auth())).status_code == 200
    await get_event_writer().flush()
    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        ev = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert (ev.status, ev.failed_message) == ("failed", "matrix bug")


async def test_success_with_no_results_in_storage_is_refused_and_fails_the_run(
    client, db, make_run, make_worker, results_in_storage
):
    results_in_storage["keys"] = []
    _, rid = await claimed_run(client, db, make_run, make_worker)
    r = await client.post(f"/api/worker/runs/{rid}/complete", json={"status": "succeeded"}, headers=auth())
    assert r.status_code == 409
    await get_event_writer().flush()
    run = await run_row(db, rid)
    assert run.status == "failed" and "no results" in run.failed_message


async def test_failed_and_canceled_outcomes_end_the_run_with_their_message(client, db, make_run, make_worker):
    _, failed = await claimed_run(client, db, make_run, make_worker)
    body = {"status": "failed", "message": "RuntimeError: CUDA out of memory"}
    assert (await client.post(f"/api/worker/runs/{failed}/complete", json=body, headers=auth())).status_code == 200

    rid2 = await make_run()
    assert (await client.post("/api/worker/claim", json=CLAIM, headers=auth())).json()["id"] == str(rid2)
    assert (
        await client.post(f"/api/worker/runs/{rid2}/complete", json={"status": "canceled"}, headers=auth())
    ).status_code == 200
    await get_event_writer().flush()

    run = await run_row(db, failed)
    assert run.status == "failed" and run.failed_message == "RuntimeError: CUDA out of memory"
    assert (await run_row(db, rid2)).status == "canceled"


async def test_completing_twice_keeps_the_first_outcome(client, db, make_run, make_worker):
    _, rid = await claimed_run(client, db, make_run, make_worker)
    first = await client.post(
        f"/api/worker/runs/{rid}/complete", json={"status": "failed", "message": "x"}, headers=auth()
    )
    await get_event_writer().flush()
    again = await client.post(f"/api/worker/runs/{rid}/complete", json={"status": "succeeded"}, headers=auth())
    assert first.json() == {"status": "failed"} and again.json() == {"status": "failed"}
    assert (await run_row(db, rid)).status == "failed"
