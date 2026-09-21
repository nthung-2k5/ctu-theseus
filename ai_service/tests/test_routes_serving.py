"""Inference (session + API-key surfaces) and export over HTTP."""

import asyncio
import json
import uuid

import pytest
import sqlalchemy as sa

from theseus.auth import rate_limit
from theseus.db.models import DatasetVersion, InferenceJob, ModelExport, TrainingRun
from theseus.jobs import inference as inference_jobs
from theseus.jobs import queue
from theseus.jobs.dispatcher import Dispatcher, Lane, set_dispatcher
from theseus.routers import inference as inference_router
from theseus.services import inference as inference_service
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


async def job_row(db, inference_id) -> InferenceJob:
    async with db() as s:
        return (await s.execute(sa.select(InferenceJob).where(InferenceJob.id == uuid.UUID(inference_id)))).scalar_one()


def fields(**kw):
    return {"fields": json.dumps(kw)}


# -- Session inference: dispatch -------------------------------------------------------------


async def test_a_text_request_creates_a_pending_job_carrying_its_payload_and_no_upload(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}", data={**fields(text="great movie"), "topK": "3"})
    assert r.status_code == 202
    j = await job_row(db, r.json()["inferenceId"])
    assert (j.status, j.top_k, j.upload_key) == ("pending", 3, None)
    assert j.payload == {"kind": "text", "fields": {"text": "great movie"}}  # extra client keys never reach the model
    polled = (await c.get(f"/api/inference/{run}/jobs/{j.id}")).json()
    assert polled == {"status": "pending"}


async def test_only_a_succeeded_run_can_be_used(client, new_user, db):
    c = await new_user()
    for status in ("queued", "running", "failed", "canceled"):
        run = await succeeded_run(c, db, status=status)
        r = await c.post(f"/api/inference/{run}", data=fields(text="x"))
        assert r.status_code == 409 and "No successfully trained model" in r.json()["error"]["message"]


async def test_field_validation_errors_are_422_with_specific_messages(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    url = f"/api/inference/{run}"
    assert "requires a `fields`" in (await c.post(url)).json()["error"]["message"]
    assert "JSON-encoded object" in (await c.post(url, data={"fields": "not json"})).json()["error"]["message"]
    assert "JSON-encoded object" in (await c.post(url, data={"fields": "[1,2]"})).json()["error"]["message"]
    assert "Missing required field(s): text" in (await c.post(url, data=fields(other="x"))).json()["error"]["message"]
    assert (await c.post(url, data=fields(text=""))).status_code == 422
    assert (await c.post(url, data={**fields(text="x"), "topK": "0"})).status_code == 422
    assert (await c.post(url, data={**fields(text="x"), "topK": "1001"})).status_code == 422


async def test_a_multi_input_task_needs_every_input_field(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db, task="question_answering")
    r = await c.post(f"/api/inference/{run}", data=fields(context="c"))
    assert r.status_code == 422 and "question" in r.json()["error"]["message"]
    assert (await c.post(f"/api/inference/{run}", data=fields(context="c", question="q"))).status_code == 202


async def test_a_tabular_record_may_only_hold_strings_and_numbers(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db, task="tabular_classification")
    ok = await c.post(f"/api/inference/{run}", data=fields(age=30, city="Hue", score=1.5))
    assert ok.status_code == 202
    assert (await job_row(db, ok.json()["inferenceId"])).payload == {
        "kind": "record",
        "record": {"age": 30, "city": "Hue", "score": 1.5},
    }
    for bad in ({"a": True}, {"a": None}, {"a": [1]}, {"a": {"b": 1}}):
        assert (await c.post(f"/api/inference/{run}", data={"fields": json.dumps(bad)})).status_code == 422


async def test_a_file_request_is_type_checked_and_stored_durably_in_the_uploads_bucket(client, new_user, db, fake_s3):
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    url = f"/api/inference/{run}"
    assert "requires a `file`" in (await c.post(url)).json()["error"]["message"]
    bad = await c.post(url, files={"file": ("x.gif", b"GIF", "image/gif")})
    assert bad.status_code == 422 and "only accepts: image/jpeg, image/png" in bad.json()["error"]["message"]

    ok = await c.post(url, files={"file": ("cat.PNG", b"\x89PNG-bytes", "image/png")})
    assert ok.status_code == 202
    j = await job_row(db, ok.json()["inferenceId"])
    assert j.upload_key == f"inference/{j.id}/input.PNG"
    assert fake_s3[("theseus-uploads", j.upload_key)] == b"\x89PNG-bytes"
    assert j.payload["kind"] == "file" and "localPath" not in j.payload  # async: durable, no temp file


async def test_an_oversized_upload_is_413_and_stores_nothing(client, new_user, db, fake_s3, monkeypatch):
    monkeypatch.setattr(inference_service, "MAX_UPLOAD_BYTES", 10)
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    r = await c.post(f"/api/inference/{run}", files={"file": ("big.png", b"x" * 11, "image/png")})
    assert r.status_code == 413
    assert fake_s3 == {}
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(InferenceJob))).scalar_one() == 0


async def test_batch_is_csv_only_for_text_and_tabular_tasks(client, new_user, db, fake_s3):
    c = await new_user()
    text_run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{text_run}/batch", files={"file": ("rows.csv", b"text\nhi\n", "text/csv")})
    assert r.status_code == 202
    j = await job_row(db, r.json()["inferenceId"])
    assert j.payload["kind"] == "batch" and j.upload_key == f"inference/{j.id}/input.csv"
    assert fake_s3[("theseus-uploads", j.upload_key)] == b"text\nhi\n"

    vision = await succeeded_run(c, db, task="image_classification")
    r = await c.post(f"/api/inference/{vision}/batch", files={"file": ("rows.csv", b"a\n1\n", "text/csv")})
    assert r.status_code == 422 and "only available for text and tabular" in r.json()["error"]["message"]
    assert (await c.post(f"/api/inference/{text_run}/batch")).status_code == 422  # file is required


# -- Session inference: reading results ------------------------------------------------------


async def test_polling_returns_each_terminal_shape_and_never_leaks_internal_keys(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    run_id = uuid.UUID(run)
    async with db() as s:
        ok = InferenceJob(run_id=run_id, status="success", payload={}, output={"kind": "classification", "classes": []})
        failed = InferenceJob(
            run_id=run_id,
            status="failed",
            payload={},
            error="Inference failed after multiple attempts.",
            last_error="s3://secret/key",
        )
        batch = InferenceJob(
            run_id=run_id,
            status="success",
            payload={},
            output={"kind": "batch", "resultKey": "r/predictions/x.csv", "rowCount": 7},
        )
        running = InferenceJob(run_id=run_id, status="running", payload={}, attempt=1)
        s.add_all([ok, failed, batch, running])
        await s.commit()
        ids = [ok.id, failed.id, batch.id, running.id]

    poll = lambda i: c.get(f"/api/inference/{run}/jobs/{i}")  # noqa: E731
    assert (await poll(ids[0])).json() == {"status": "success", "output": {"kind": "classification", "classes": []}}
    assert (await poll(ids[1])).json() == {"status": "failed", "error": "Inference failed after multiple attempts."}
    assert (await poll(ids[2])).json() == {"status": "batch", "rowCount": 7}  # resultKey is internal
    assert (await poll(ids[3])).json() == {"status": "pending"}  # `running` is not a client-visible state
    assert "secret" not in (await poll(ids[1])).text


async def test_a_job_id_from_another_run_or_user_cannot_be_read(client, new_user, db):
    owner, stranger = await new_user(), await new_user()
    mine, theirs = await succeeded_run(owner, db), await succeeded_run(stranger, db)
    r = await owner.post(f"/api/inference/{mine}", data=fields(text="x"))
    jid = r.json()["inferenceId"]
    assert (await stranger.get(f"/api/inference/{mine}/jobs/{jid}")).status_code == 403
    assert (
        await stranger.get(f"/api/inference/{theirs}/jobs/{jid}")
    ).status_code == 404  # scoped to the run in the query
    assert (await owner.get(f"/api/inference/{mine}/jobs/{uuid.uuid4()}")).status_code == 404
    assert (await stranger.post(f"/api/inference/{mine}", data=fields(text="x"))).status_code == 403


async def test_history_is_newest_first_and_hides_the_internal_running_state(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    a = (await c.post(f"/api/inference/{run}", data=fields(text="one"))).json()["inferenceId"]
    b = (await c.post(f"/api/inference/{run}", data=fields(text="two"))).json()["inferenceId"]
    async with db() as s:
        await s.execute(
            sa.update(InferenceJob).where(InferenceJob.id == uuid.UUID(b)).values(status="running", attempt=1)
        )
        await s.commit()
    jobs = (await c.get(f"/api/inference/{run}/jobs")).json()["jobs"]
    assert [j["id"] for j in jobs] == [b, a]
    assert [j["status"] for j in jobs] == ["pending", "pending"]


async def test_batch_download_redirects_to_the_result_and_is_404_otherwise(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    async with db() as s:
        done = InferenceJob(
            run_id=uuid.UUID(run),
            status="success",
            payload={},
            output={"kind": "batch", "resultKey": f"{run}/predictions/z.csv", "rowCount": 1},
        )
        single = InferenceJob(run_id=uuid.UUID(run), status="success", payload={}, output={"kind": "text"})
        s.add_all([done, single])
        await s.commit()
    r = await c.get(f"/api/inference/{run}/jobs/{done.id}/download", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == f"https://s3.test/theseus-models/{run}/predictions/z.csv"
    assert (await c.get(f"/api/inference/{run}/jobs/{single.id}/download")).status_code == 404


async def test_warm_requires_a_succeeded_run_and_preloads_in_the_background(client, new_user, db, monkeypatch):
    warmed: list[str] = []
    monkeypatch.setattr(inference_router, "spawn_warm", warmed.append)
    c = await new_user()
    assert (await c.post(f"/api/inference/{await succeeded_run(c, db, status='running')}/warm")).status_code == 409
    run = await succeeded_run(c, db)
    assert (await c.post(f"/api/inference/{run}/warm")).status_code == 202
    assert warmed == [run]


# -- API-key surface (/api/v1/predict) -------------------------------------------------------


async def make_key(c) -> str:
    return (await c.post("/api/keys", json={"name": "k"})).json()["key"]


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


async def test_the_api_key_surface_requires_a_valid_unrevoked_key_for_a_run_the_key_owner_owns(client, new_user, db):
    owner, other = await new_user(), await new_user()
    run = await succeeded_run(owner, db)
    key, other_key = await make_key(owner), await make_key(other)
    url = f"/api/v1/predict/{run}"

    assert (await client.post(url, data=fields(text="x"))).status_code == 401  # no header
    assert (await client.post(url, data=fields(text="x"), headers=bearer("thsk_nope"))).status_code == 401
    assert (await client.post(url, data=fields(text="x"), headers={"Authorization": f"Basic {key}"})).status_code == 401
    assert (
        await client.post(url, data=fields(text="x"), headers=bearer(other_key))
    ).status_code == 403  # valid key, not their run
    assert (
        await client.post(f"/api/v1/predict/{uuid.uuid4()}", data=fields(text="x"), headers=bearer(key))
    ).status_code == 404

    ok = await client.post(url, data=fields(text="x"), headers=bearer(key))
    assert ok.status_code == 202 and ok.headers["x-ratelimit-limit"] == "60"
    # a session cookie does NOT authenticate the v1 surface (and a bearer key does not authenticate /api)
    assert (await owner.post(url, data=fields(text="x"))).status_code == 401
    assert (await client.get(f"/api/runs/{run}", headers=bearer(key))).status_code == 401


async def test_a_revoked_key_stops_working_and_last_used_is_recorded(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    created = (await c.post("/api/keys", json={"name": "k"})).json()
    await client.post(f"/api/v1/predict/{run}", data=fields(text="x"), headers=bearer(created["key"]))
    for _ in range(20):
        used = [k for k in (await c.get("/api/keys")).json()["keys"] if k["id"] == created["id"]][0]["lastUsedAt"]
        if used:
            break
        await asyncio.sleep(0.05)
    assert used is not None

    await c.delete(f"/api/keys/{created['id']}")
    assert (
        await client.post(f"/api/v1/predict/{run}", data=fields(text="x"), headers=bearer(created["key"]))
    ).status_code == 401


async def test_the_rate_limit_is_per_key_with_retry_after_and_applies_before_the_database(
    client, new_user, db, monkeypatch
):
    from theseus import deps

    monkeypatch.setattr(deps, "API_KEY_RATE_LIMIT_MAX", 2)
    c = await new_user()
    run = await succeeded_run(c, db)
    key, other = await make_key(c), await make_key(c)
    call = lambda k: client.get(f"/api/v1/predict/{run}/jobs/{uuid.uuid4()}", headers=bearer(k))  # noqa: E731
    assert [(await call(key)).status_code for _ in range(2)] == [404, 404]
    limited = await call(key)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert limited.headers["x-ratelimit-remaining"] == "0"
    assert (await call(other)).status_code == 404  # another key is unaffected
    # an INVALID key is throttled too, before any DB lookup, so guessing cannot hammer Postgres
    codes = [
        (await client.get(f"/api/v1/predict/{run}/jobs/{uuid.uuid4()}", headers=bearer("thsk_guess"))).status_code
        for _ in range(3)
    ]
    assert codes == [401, 401, 429]


async def test_v1_is_exempt_from_the_cookie_csrf_origin_check_because_it_uses_bearer_auth(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    r = await client.post(
        f"/api/v1/predict/{run}", data=fields(text="x"), headers={**bearer(key), "Origin": "https://partner.example"}
    )
    assert r.status_code == 202
    blocked = await c.post(f"/api/inference/{run}", data=fields(text="x"), headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403  # the cookie surface still enforces it


async def test_v1_poll_shares_the_same_shapes_and_run_scoping(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    jid = (await client.post(f"/api/v1/predict/{run}", data=fields(text="x"), headers=bearer(key))).json()[
        "inferenceId"
    ]
    assert (await client.get(f"/api/v1/predict/{run}/jobs/{jid}", headers=bearer(key))).json() == {"status": "pending"}
    assert (await client.get(f"/api/v1/predict/{run}/jobs/{uuid.uuid4()}", headers=bearer(key))).status_code == 404


async def _finish_when_queued(db, output):
    """Play the worker: wait for the job row, mark it succeeded, wake the waiting request."""
    for _ in range(100):
        async with db() as s:
            job = (await s.execute(sa.select(InferenceJob))).scalars().first()
        if job is not None:
            async with db() as s:
                await s.execute(
                    sa.update(InferenceJob).where(InferenceJob.id == job.id).values(status="success", output=output)
                )
                await s.commit()
            inference_jobs._resolve_waiter(job.id, "success")
            return job.id
        await asyncio.sleep(0.02)


async def test_sync_predict_is_a_real_await_that_returns_the_result_inline(client, new_user, db, monkeypatch):
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    worker = asyncio.create_task(_finish_when_queued(db, {"kind": "classification", "feature": "class", "classes": []}))
    r = await client.post(f"/api/v1/predict/{run}/sync", data=fields(text="x"), headers=bearer(key))
    job_id = await worker
    assert r.status_code == 200
    assert r.json() == {
        "inferenceId": str(job_id),
        "status": "success",
        "output": {"kind": "classification", "feature": "class", "classes": []},
    }


async def test_sync_predict_falls_back_to_202_on_timeout_and_the_job_keeps_running(client, new_user, db, monkeypatch):
    from theseus.settings import get_settings

    monkeypatch.setattr(get_settings(), "sync_predict_max_wait_seconds", 0.2)
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    r = await client.post(f"/api/v1/predict/{run}/sync", data=fields(text="x"), headers=bearer(key))
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending" and set(body) == {"inferenceId", "status"}
    j = await job_row(db, body["inferenceId"])
    assert j.status == "pending"  # NOT cancelled by the timeout: the caller can poll it
    assert uuid.UUID(body["inferenceId"]) not in inference_jobs._waiters  # and no waiter is leaked


async def test_sync_predict_is_503_with_retry_after_when_every_inference_slot_is_busy(client, new_user, db):
    async def noop(job_id):
        pass

    lane = Lane("inference", queue.INFERENCE, 1, noop)
    lane.running.add(uuid.uuid4())  # the only slot is taken
    set_dispatcher(Dispatcher([lane]))
    try:
        c = await new_user()
        run = await succeeded_run(c, db)
        r = await client.post(f"/api/v1/predict/{run}/sync", data=fields(text="x"), headers=bearer(await make_key(c)))
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        async with db() as s:
            assert (
                await s.execute(sa.select(sa.func.count()).select_from(InferenceJob))
            ).scalar_one() == 0  # nothing queued
    finally:
        set_dispatcher(None)


async def test_a_synchronous_file_request_keeps_its_upload_in_a_temp_file_not_s3(
    client, new_user, db, fake_s3, monkeypatch, tmp_path
):
    from theseus.settings import get_settings

    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    monkeypatch.setattr(get_settings(), "sync_predict_max_wait_seconds", 0.1)
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    r = await client.post(
        f"/api/v1/predict/{run}/sync",
        files={"file": ("a.png", b"png-bytes", "image/png")},
        headers=bearer(await make_key(c)),
    )
    assert r.status_code == 202
    j = await job_row(db, r.json()["inferenceId"])
    assert j.upload_key is None and fake_s3 == {}  # never touched S3
    assert open(j.payload["localPath"], "rb").read() == b"png-bytes"


# -- Export ----------------------------------------------------------------------------------


async def test_export_validation_rules_per_tier(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    url = f"/api/runs/{run}/exports"
    msg = lambda r: r.json()["error"]["message"]  # noqa: E731
    assert (await c.post(url, json={"tier": "model", "format": "torchscript"})).status_code == 202
    assert "only supports format 'onnx'" in msg(
        await c.post(url, json={"tier": "devkit", "format": "torchscript", "lang": "python"})
    )
    assert "requires a lang" in msg(await c.post(url, json={"tier": "devkit", "format": "onnx"}))
    assert "one of: python, typescript, csharp, java" in msg(
        await c.post(url, json={"tier": "devkit", "format": "onnx", "lang": "pwa"})
    )
    assert "one of: pwa, flutter" in msg(await c.post(url, json={"tier": "app", "format": "onnx", "lang": "python"}))
    assert (await c.post(url, json={"tier": "app", "format": "onnx", "lang": "flutter"})).status_code == 202
    assert (await c.post(url, json={"tier": "nope", "format": "onnx"})).status_code == 422
    unfinished = await succeeded_run(c, db, status="running")
    assert (
        await c.post(f"/api/runs/{unfinished}/exports", json={"tier": "model", "format": "onnx"})
    ).status_code == 409


async def test_an_export_starts_pending_owned_by_the_project_owner_and_lists_newest_first(client, new_user, db):
    c = await new_user()
    run = await succeeded_run(c, db)
    a = (await c.post(f"/api/runs/{run}/exports", json={"tier": "model", "format": "onnx", "lang": "python"})).json()[
        "exportId"
    ]
    b = (await c.post(f"/api/runs/{run}/exports", json={"tier": "devkit", "format": "onnx", "lang": "python"})).json()[
        "exportId"
    ]
    async with db() as s:
        rows = {str(e.id): e for e in (await s.execute(sa.select(ModelExport))).scalars()}
    assert rows[a].lang is None  # the model tier never has a language, even if the client sent one
    assert rows[b].lang == "python" and rows[a].status == "pending" and str(rows[a].user_id) == c.user_id
    assert rows[a].max_attempts == 3

    listed = (await c.get(f"/api/runs/{run}/exports")).json()["exports"]
    assert [e["id"] for e in listed] == [b, a]
    got = (await c.get(f"/api/exports/{a}")).json()["export"]
    assert got["status"] == "pending" and got["tier"] == "model"


async def test_export_download_is_404_until_ready_then_redirects_and_is_owner_only(client, new_user, db):
    owner, stranger = await new_user(), await new_user()
    run = await succeeded_run(owner, db)
    eid = (await owner.post(f"/api/runs/{run}/exports", json={"tier": "model", "format": "onnx"})).json()["exportId"]
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
    assert (
        await stranger.post(f"/api/runs/{run}/exports", json={"tier": "model", "format": "onnx"})
    ).status_code == 403
