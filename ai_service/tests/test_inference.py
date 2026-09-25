"""Inference over HTTP: predictions run inside the request and nothing is stored.

Covers both surfaces (session cookie /api/inference and API key /api/v1/predict) with a fake model
standing in for the trained one; the prediction shaping itself is tested in test_ludwig_model.py.
"""

import asyncio
import json
import os
import time
import uuid

import pandas as pd
import pytest
import sqlalchemy as sa

from theseus.auth import rate_limit
from theseus.db.models import DatasetVersion, TrainingRun
from theseus.routers import inference as inference_router
from theseus.services import inference as svc
from theseus.settings import get_settings


class FakeModel:
    predict_calls: list[pd.DataFrame] = []
    outputs: list[dict] = []
    delay = 0.0
    fail_with: Exception | None = None
    file_existed_during_predict: list[bool] = []

    def __init__(self, columns=("text",)):
        self.input_columns = list(columns)

    def predict(self, frame):
        FakeModel.predict_calls.append(frame)
        for col in self.input_columns:
            value = frame[col].iloc[0]
            if isinstance(value, str) and os.sep in value:
                FakeModel.file_existed_during_predict.append(os.path.exists(value))
        time.sleep(FakeModel.delay)
        if FakeModel.fail_with is not None:
            raise FakeModel.fail_with
        return pd.DataFrame({"class_predictions": ["cat"] * len(frame)})

    def to_output(self, predictions, *, top_k=100, input_tokens=None):
        out = {
            "kind": "classification",
            "feature": "class",
            "classes": [{"label": "cat", "confidence": 0.9}],
            "topK": top_k,
            "inputTokens": input_tokens,
        }
        FakeModel.outputs.append(out)
        return out


@pytest.fixture
def model(monkeypatch):
    FakeModel.predict_calls, FakeModel.outputs = [], []
    FakeModel.delay, FakeModel.fail_with, FakeModel.file_existed_during_predict = 0.0, None, []
    box = {"model": FakeModel()}

    class Cache:
        async def get(self, run_id):
            return box["model"]

    monkeypatch.setattr(svc, "_model_cache", lambda: Cache())
    rate_limit.reset()
    return box


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


def fields(**kw):
    return {"fields": json.dumps(kw)}


def msg(r):
    return r.json()["error"]["message"]


# -- Single prediction -----------------------------------------------------------------------


async def test_a_text_request_returns_the_prediction_in_the_response(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}", data={**fields(text="great movie"), "topK": "3"})
    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["kind"] == "classification" and out["classes"] == [{"label": "cat", "confidence": 0.9}]
    assert out["topK"] == 3
    assert out["inputTokens"] == ["great", "movie"]
    assert FakeModel.predict_calls[0].to_dict("records") == [{"text": "great movie"}]


async def test_nothing_is_stored_for_a_prediction(client, new_user, db, model):
    from theseus.db.base import Base

    assert "inference_jobs" not in Base.metadata.tables
    c = await new_user()
    run = await succeeded_run(c, db)
    assert (await c.post(f"/api/inference/{run}", data=fields(text="x"))).status_code == 200
    async with db() as s:
        tables = (
            await s.execute(sa.text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        ).scalars()
        assert "inference_jobs" not in set(tables)


async def test_only_a_succeeded_run_can_be_used(client, new_user, db, model):
    c = await new_user()
    for status in ("queued", "running", "failed", "canceled"):
        run = await succeeded_run(c, db, status=status)
        r = await c.post(f"/api/inference/{run}", data=fields(text="x"))
        assert r.status_code == 409 and "No successfully trained model" in msg(r)
    assert FakeModel.predict_calls == []


async def test_field_validation_errors_are_422_with_specific_messages(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    url = f"/api/inference/{run}"
    assert "requires a `fields`" in msg(await c.post(url))
    assert "JSON-encoded object" in msg(await c.post(url, data={"fields": "not json"}))
    assert "JSON-encoded object" in msg(await c.post(url, data={"fields": "[1,2]"}))
    assert "Missing required field(s): text" in msg(await c.post(url, data=fields(other="x")))
    assert (await c.post(url, data=fields(text=""))).status_code == 422
    assert (await c.post(url, data={**fields(text="x"), "topK": "0"})).status_code == 422
    assert (await c.post(url, data={**fields(text="x"), "topK": "1001"})).status_code == 422
    assert FakeModel.predict_calls == []  # rejected before the model was touched


async def test_a_multi_input_task_needs_every_input_field(client, new_user, db, model):
    model["model"] = FakeModel(columns=("context", "question"))
    c = await new_user()
    run = await succeeded_run(c, db, task="question_answering")
    r = await c.post(f"/api/inference/{run}", data=fields(context="c"))
    assert r.status_code == 422 and "question" in msg(r)
    ok = await c.post(f"/api/inference/{run}", data=fields(context="c", question="q"))
    assert ok.status_code == 200
    assert FakeModel.predict_calls[0].to_dict("records") == [{"context": "c", "question": "q"}]


async def test_the_model_can_reject_a_request_the_task_spec_accepted(client, new_user, db, model):
    model["model"] = FakeModel(columns=("a", "b"))  # the trained model wants columns the spec did not know about
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}", data=fields(text="x"))
    assert r.status_code == 422 and "Missing required field(s): a, b" in msg(r)


async def test_a_tabular_record_may_only_hold_strings_and_numbers(client, new_user, db, model):
    model["model"] = FakeModel(columns=("age", "city"))
    c = await new_user()
    run = await succeeded_run(c, db, task="tabular_classification")
    ok = await c.post(f"/api/inference/{run}", data=fields(age=30, city="Hue", score=1.5))
    assert ok.status_code == 200
    assert FakeModel.predict_calls[0].to_dict("records") == [{"age": 30, "city": "Hue", "score": 1.5}]
    for bad in ({"a": True}, {"a": None}, {"a": [1]}, {"a": {"b": 1}}):
        assert (await c.post(f"/api/inference/{run}", data={"fields": json.dumps(bad)})).status_code == 422


async def test_a_file_request_is_type_checked_and_scored_from_a_temp_file_that_is_removed(client, new_user, db, model):
    model["model"] = FakeModel(columns=("image_path",))
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    url = f"/api/inference/{run}"
    assert "requires a `file`" in msg(await c.post(url))
    bad = await c.post(url, files={"file": ("x.gif", b"GIF", "image/gif")})
    assert bad.status_code == 422 and "only accepts: image/jpeg, image/png" in msg(bad)

    ok = await c.post(url, files={"file": ("cat.PNG", b"\x89PNG-bytes", "image/png")})
    assert ok.status_code == 200, ok.text
    path = FakeModel.predict_calls[0]["image_path"].iloc[0]
    assert path.endswith("input.PNG")
    assert FakeModel.file_existed_during_predict == [True]  # there while predicting...
    assert not os.path.exists(path)  # ...gone afterwards


async def test_an_oversized_upload_is_413_before_the_model_is_touched(client, new_user, db, model, monkeypatch):
    monkeypatch.setattr(svc, "MAX_UPLOAD_BYTES", 10)
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    r = await c.post(f"/api/inference/{run}", files={"file": ("big.png", b"x" * 11, "image/png")})
    assert r.status_code == 413
    assert FakeModel.predict_calls == []


async def test_an_upload_extension_is_sanitized(client, new_user, db, model):
    model["model"] = FakeModel(columns=("image_path",))
    c = await new_user()
    run = await succeeded_run(c, db, task="image_classification")
    r = await c.post(f"/api/inference/{run}", files={"file": ("a.png/../../etc/x.p ng", b"1", "image/png")})
    assert r.status_code == 200
    assert os.path.basename(FakeModel.predict_calls[0]["image_path"].iloc[0]).startswith("input")


# -- Failure, timeout, concurrency -----------------------------------------------------------


async def test_an_unexpected_failure_is_a_500_that_hides_internals(client, new_user, db, model):
    FakeModel.fail_with = RuntimeError("s3://theseus-models/secret/key: AccessDenied")
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}", data=fields(text="x"))
    assert r.status_code == 500
    assert msg(r) == svc.GENERIC_FAILURE and "secret" not in r.text


async def test_a_prediction_that_exceeds_the_timeout_is_a_504_and_its_slot_frees_when_the_thread_ends(
    client, new_user, db, model, monkeypatch
):
    monkeypatch.setattr(get_settings(), "inference_timeout_seconds", 0.1)
    monkeypatch.setattr(get_settings(), "inference_concurrency", 1)
    monkeypatch.setattr(svc, "BUSY_WAIT_SECONDS", 0.05)
    FakeModel.delay = 0.6
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}", data=fields(text="x"))
    assert r.status_code == 504

    # The thread cannot be interrupted, so the only slot is still held until it really finishes.
    busy = await c.post(f"/api/inference/{run}", data=fields(text="x"))
    assert busy.status_code == 503

    await asyncio.sleep(0.8)
    FakeModel.delay = 0.0
    monkeypatch.setattr(get_settings(), "inference_timeout_seconds", 5)
    assert (await c.post(f"/api/inference/{run}", data=fields(text="x"))).status_code == 200


async def test_when_every_slot_is_busy_the_request_is_a_503_with_retry_after(client, new_user, db, model, monkeypatch):
    monkeypatch.setattr(get_settings(), "inference_concurrency", 1)
    monkeypatch.setattr(svc, "BUSY_WAIT_SECONDS", 0.05)
    FakeModel.delay = 0.4
    c = await new_user()
    run = await succeeded_run(c, db)
    slow = asyncio.create_task(c.post(f"/api/inference/{run}", data=fields(text="slow")))
    await asyncio.sleep(0.1)  # let it take the slot
    r = await c.post(f"/api/inference/{run}", data=fields(text="fast"))
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert (await slow).status_code == 200  # the running prediction is unaffected


async def test_a_stranger_cannot_use_someone_elses_run(client, new_user, db, model):
    owner, stranger = await new_user(), await new_user()
    run = await succeeded_run(owner, db)
    assert (await stranger.post(f"/api/inference/{run}", data=fields(text="x"))).status_code == 403
    assert (await stranger.post(f"/api/inference/{uuid.uuid4()}", data=fields(text="x"))).status_code == 404
    assert FakeModel.predict_calls == []


async def test_warm_requires_a_succeeded_run_and_preloads_in_the_background(client, new_user, db, monkeypatch):
    warmed: list[str] = []
    monkeypatch.setattr(inference_router, "spawn_warm", warmed.append)
    c = await new_user()
    assert (await c.post(f"/api/inference/{await succeeded_run(c, db, status='running')}/warm")).status_code == 409
    run = await succeeded_run(c, db)
    assert (await c.post(f"/api/inference/{run}/warm")).status_code == 202
    assert warmed == [run]


# -- Batch -----------------------------------------------------------------------------------


def csv_file(text):
    return {"file": ("rows.csv", text.encode(), "text/csv")}


async def test_batch_scores_every_row_in_one_call_and_returns_the_csv(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await c.post(f"/api/inference/{run}/batch", files=csv_file("text\nhi\nthere\n"))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv") and r.headers["x-row-count"] == "2"
    assert "attachment" in r.headers["content-disposition"]
    rows = r.text.strip().splitlines()
    assert rows[0] == "text,class_predictions" and len(rows) == 3
    assert len(FakeModel.predict_calls) == 1  # one predict call for the whole file, not one per row


async def test_batch_rejects_bad_input_with_clear_messages(client, new_user, db, model, monkeypatch):
    c = await new_user()
    run = await succeeded_run(c, db)
    url = f"/api/inference/{run}/batch"
    assert (await c.post(url)).status_code == 422  # file is required
    assert "no rows" in msg(await c.post(url, files=csv_file("text\n")))
    assert "not a readable CSV" in msg(await c.post(url, files=csv_file("")))
    monkeypatch.setattr(svc, "MAX_BATCH_ROWS", 1)
    over = await c.post(url, files=csv_file("text\na\nb\n"))
    assert over.status_code == 422 and "exceeding the 1-row limit" in msg(over)
    assert FakeModel.predict_calls == []


async def test_batch_is_only_for_text_and_tabular_tasks(client, new_user, db, model):
    c = await new_user()
    vision = await succeeded_run(c, db, task="image_classification")
    r = await c.post(f"/api/inference/{vision}/batch", files=csv_file("a\n1\n"))
    assert r.status_code == 422 and "only available for text and tabular" in msg(r)


# -- API-key surface (/api/v1/predict) -------------------------------------------------------


async def make_key(c) -> str:
    return (await c.post("/api/keys", json={"name": "k"})).json()["key"]


def bearer(key):
    return {"Authorization": f"Bearer {key}"}


async def test_the_api_key_surface_requires_a_valid_unrevoked_key_for_a_run_the_key_owner_owns(
    client, new_user, db, model
):
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
    assert ok.status_code == 200 and ok.headers["x-ratelimit-limit"] == "60"
    assert ok.json()["output"]["kind"] == "classification"
    # a session cookie does NOT authenticate the v1 surface (and a bearer key does not authenticate /api)
    assert (await owner.post(url, data=fields(text="x"))).status_code == 401
    assert (await client.get(f"/api/runs/{run}", headers=bearer(key))).status_code == 401


async def test_the_old_sync_path_still_answers_the_same_way(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await client.post(f"/api/v1/predict/{run}/sync", data=fields(text="x"), headers=bearer(await make_key(c)))
    assert r.status_code == 200 and r.json()["output"]["kind"] == "classification"


async def test_v1_batch_returns_the_scored_csv(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    r = await client.post(
        f"/api/v1/predict/{run}/batch", files=csv_file("text\nhi\n"), headers=bearer(await make_key(c))
    )
    assert r.status_code == 200 and r.headers["x-row-count"] == "1" and "class_predictions" in r.text


async def test_the_polling_and_history_routes_are_gone(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    job = uuid.uuid4()
    assert (await c.get(f"/api/inference/{run}/jobs")).status_code in (404, 405)
    assert (await c.get(f"/api/inference/{run}/jobs/{job}")).status_code in (404, 405)
    assert (await client.get(f"/api/v1/predict/{run}/jobs/{job}", headers=bearer(key))).status_code in (404, 405)


async def test_a_revoked_key_stops_working_and_last_used_is_recorded(client, new_user, db, model):
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
    client, new_user, db, model, monkeypatch
):
    from theseus import deps

    monkeypatch.setattr(deps, "API_KEY_RATE_LIMIT_MAX", 2)
    c = await new_user()
    run = await succeeded_run(c, db)
    key, other = await make_key(c), await make_key(c)
    call = lambda k: client.post(f"/api/v1/predict/{run}", data=fields(text="x"), headers=bearer(k))  # noqa: E731
    assert [(await call(key)).status_code for _ in range(2)] == [200, 200]
    limited = await call(key)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert limited.headers["x-ratelimit-remaining"] == "0"
    assert (await call(other)).status_code == 200  # another key is unaffected
    # an INVALID key is throttled too, before any DB lookup, so guessing cannot hammer Postgres
    codes = [(await call("thsk_guess")).status_code for _ in range(3)]
    assert codes == [401, 401, 429]


async def test_v1_is_exempt_from_the_cookie_csrf_origin_check_because_it_uses_bearer_auth(client, new_user, db, model):
    c = await new_user()
    run = await succeeded_run(c, db)
    key = await make_key(c)
    r = await client.post(
        f"/api/v1/predict/{run}", data=fields(text="x"), headers={**bearer(key), "Origin": "https://partner.example"}
    )
    assert r.status_code == 200
    blocked = await c.post(f"/api/inference/{run}", data=fields(text="x"), headers={"Origin": "https://evil.example"})
    assert blocked.status_code == 403  # the cookie surface still enforces it
