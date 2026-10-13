"""The remote worker end to end: the real worker loop (`omnitrain.worker`) in a thread, talking HTTP to the real app and
Postgres, training a FAKE backend through `jobs.run_core`. Only S3 is replaced.

The worker is synchronous and the app is async, so the worker's HTTP client is given a transport that hands each request
to the app on the test's event loop.
"""

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
import sqlalchemy as sa

from omnitrain.backends.base import EvalResult, LoadedModel, OutputSpec, TrainContext, TrainerBackend
from omnitrain.backends.base import _registry as backend_registry
from omnitrain.db.models import CustomModel, RunEvaluation, TrainingMetric, TrainingRun
from omnitrain.events import get_event_writer
from omnitrain.jobs import abort, run_core
from omnitrain.services import custom_models, storage
from omnitrain.settings import get_settings
from omnitrain.worker.client import ApiClient
from omnitrain.worker.config import WorkerConfig
from omnitrain.worker.runner import serve

TOKEN = "thwk_loop-test"


class FakeLoadedModel(LoadedModel):
    input_columns = ["x"]
    output = OutputSpec(name="class", kind="classification")

    def predict(self, frame):
        return frame

    def to_output(self, predictions, *, top_k=100, input_tokens=None):
        return {}

    def golden_prediction(self, predictions, threshold=0.0):
        return {}


class FakeFrame:
    def to_parquet(self, path):
        pass


class FakeBackend(TrainerBackend):
    id = "fakeremote"
    label = "Fake remote"

    epochs = 3
    epoch_seconds = 0.0
    fail_with: Exception | None = None
    on_epoch_done = None
    trained_epochs = 0
    custom_seen: dict | None = None
    # Its trained model never points back at the base model's files, so the server may send it custom-model runs.
    remote_custom_models = True

    @classmethod
    def supports(cls, task):
        return True

    @classmethod
    def models(cls, task):
        return []

    @classmethod
    def compile(cls, task, ctx, hp):
        return {}

    @classmethod
    def load(cls, model_dir):
        raise NotImplementedError

    @classmethod
    def convert(cls, model, artifact_id, workdir):
        raise NotImplementedError

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        cls.trained_epochs = 0
        if custom := run.config.get("custom"):
            cls.custom_seen = {
                "path": custom["path"],
                "has_files": os.path.isfile(os.path.join(custom["path"], "config.json")),
            }
        for epoch in range(1, cls.epochs + 1):
            run.check_abort()
            run.heartbeat()
            if cls.fail_with:
                raise cls.fail_with
            run.report(epoch, "train", {"loss": 1.0 / epoch + 0.1, "accuracy": 0.5})
            run.report(epoch, "validation", {"loss": 1.0 / epoch})
            cls.trained_epochs = epoch
            if cls.on_epoch_done:
                cls.on_epoch_done(epoch)
            time.sleep(cls.epoch_seconds)
        return FakeLoadedModel()

    @classmethod
    def evaluate(cls, model, df, split_column, item_id_column):
        report = {"split": "test", "overall": {"accuracy": 0.9, "macroF1": 0.85}}
        return EvalResult(report=report, predictions=FakeFrame())


# Registered by the class statement itself, like every backend; the `env` fixture registers it only around each test.
backend_registry.unregister("fakeremote")


class LoopBridge(httpx.BaseTransport):
    """A synchronous transport that runs each request on the app, which lives on the test's event loop."""

    def __init__(self, client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop
        self.down = threading.Event()  # set: every request fails as if the API were unreachable

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.down.is_set():
            raise httpx.ConnectError("API unreachable", request=request)
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in ("host", "content-length")]
        body = request.read()

        async def go() -> httpx.Response:
            return await self._client.request(request.method, request.url.path, headers=headers, content=body)

        response = asyncio.run_coroutine_threadsafe(go(), self._loop).result(30)
        skip = ("content-length", "content-encoding", "transfer-encoding")
        return httpx.Response(
            response.status_code,
            headers=[(k, v) for k, v in response.headers.items() if k.lower() not in skip],
            content=response.content,
            request=request,
        )


@pytest.fixture
async def env(client, db, make_run, make_worker, monkeypatch, tmp_path):
    backend_registry.register(FakeBackend)
    FakeBackend.epochs, FakeBackend.epoch_seconds = 3, 0.0
    FakeBackend.fail_with, FakeBackend.on_epoch_done = None, None
    FakeBackend.custom_seen = None

    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    monkeypatch.setattr(run_core.pd, "read_parquet", lambda path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: True)
    monkeypatch.setattr(storage, "delete_prefix", lambda bucket, prefix: 0)
    monkeypatch.setattr(storage, "upload_json", lambda bucket, key, data: None)
    monkeypatch.setattr(storage, "list_keys", lambda bucket, prefix: ["results/model"])
    await make_worker(backends=("fakeremote",), token=TOKEN)
    bridge = LoopBridge(client, asyncio.get_running_loop())

    shutdowns: list[threading.Event] = []

    async def run_worker(max_runs: int = 1, shutdown: threading.Event | None = None, token: str = TOKEN) -> int:
        shutdown = shutdown or threading.Event()
        shutdowns.append(shutdown)
        config = WorkerConfig(
            api_url="http://test", token=token, poll_interval_seconds=0.05, flush_interval_seconds=0.05
        )
        api = ApiClient("http://test", token, transport=bridge)
        return await asyncio.wait_for(asyncio.to_thread(serve, config, api, shutdown, max_runs=max_runs), 90)

    async def queue_run(**fields) -> object:
        return await make_run(backend="fakeremote", **fields)

    yield SimpleNamespace(run_worker=run_worker, queue_run=queue_run, bridge=bridge)
    # `serve` runs in a thread that wait_for/cancel cannot stop. A worker still polling when its test ends (nothing to
    # claim, a failed assertion, a timeout) would keep the event loop's executor shutdown waiting forever: pytest passes
    # the test and then never exits. Tell every worker to stop.
    for shutdown in shutdowns:
        shutdown.set()
    backend_registry.unregister("fakeremote")


async def run_row(db, run_id):
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()


async def count(db, model):
    async with db() as s:
        return (await s.execute(sa.select(sa.func.count()).select_from(model))).scalar_one()


async def test_the_worker_trains_a_queued_run_and_the_server_records_all_of_it(db, env):
    rid = await env.queue_run()

    assert await env.run_worker() == 0
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert run.status == "succeeded" and run.best_epoch == 3
    assert run.worker_id is not None and run.claimed_by == f"worker:{run.worker_id}"
    assert run.started_at is not None and run.completed_at is not None
    assert await count(db, TrainingMetric) == 3 * 3  # per epoch: train loss + accuracy + validation loss
    async with db() as s:
        ev = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert (ev.status, ev.split) == ("success", "test")
    assert ev.accuracy == pytest.approx(0.9) and ev.macro_f1 == pytest.approx(0.85)
    assert ev.report_key == f"{rid}/evaluation/report.json"


async def test_a_training_error_on_the_worker_fails_the_run_with_its_message(db, env):
    FakeBackend.fail_with = RuntimeError("CUDA out of memory")
    rid = await env.queue_run()

    assert await env.run_worker() == 0
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert run.status == "failed" and run.failed_message == "RuntimeError: CUDA out of memory"
    assert await count(db, RunEvaluation) == 0


async def test_canceling_a_run_stops_the_remote_training_early(db, env):
    FakeBackend.epochs, FakeBackend.epoch_seconds = 500, 0.02
    epoch_one_done, resume = threading.Event(), threading.Event()

    def on_epoch_done(epoch):
        if epoch == 1:
            epoch_one_done.set()
            resume.wait(10)

    FakeBackend.on_epoch_done = on_epoch_done
    rid = await env.queue_run()
    worker = asyncio.create_task(env.run_worker())
    await asyncio.to_thread(epoch_one_done.wait, 20)

    assert await abort.request_cancel(rid) is True
    resume.set()
    assert await asyncio.wait_for(worker, 60) == 0
    await get_event_writer().flush()

    assert FakeBackend.trained_epochs < 400  # it noticed through a progress call, long before the last epoch
    run = await run_row(db, rid)
    assert run.status == "canceled" and run.cancel_requested_at is not None


async def test_an_api_outage_during_training_loses_no_progress(db, env):
    FakeBackend.epochs, FakeBackend.epoch_seconds = 6, 0.05

    def on_epoch_done(epoch):
        if epoch == 2:
            env.bridge.down.set()
        if epoch == 4:
            time.sleep(0.3)  # long enough for several failed flushes
            env.bridge.down.clear()

    FakeBackend.on_epoch_done = on_epoch_done
    rid = await env.queue_run()

    assert await env.run_worker() == 0
    await get_event_writer().flush()

    assert (await run_row(db, rid)).status == "succeeded"
    assert await count(db, TrainingMetric) == 6 * 3  # nothing dropped, nothing doubled


async def test_shutting_the_worker_down_mid_run_fails_the_run_instead_of_canceling_it(db, env):
    FakeBackend.epochs, FakeBackend.epoch_seconds = 500, 0.02
    shutdown = threading.Event()
    FakeBackend.on_epoch_done = lambda epoch: shutdown.set() if epoch == 2 else None
    rid = await env.queue_run()

    assert await env.run_worker(shutdown=shutdown) == 0
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert run.status == "failed" and "shut down" in run.failed_message


async def test_a_worker_with_a_wrong_token_stops_with_an_error_code(db, env):
    rid = await env.queue_run()
    assert await env.run_worker(token="thwk_wrong") == 1
    assert (await run_row(db, rid)).status == "queued"


async def make_custom_model(db) -> object:
    async with db() as s:
        model = CustomModel(
            backend="fakeremote", kind="hf_video", name="m", source_kind="upload", sha256="a" * 64, status="ready"
        )
        s.add(model)
        await s.commit()
        return model


async def test_a_custom_model_run_trains_on_the_worker_against_its_own_copy_of_the_model(
    db, env, monkeypatch, tmp_path
):
    model = await make_custom_model(db)
    server_path = str(custom_models.local_dir_for(model))  # under the server's temp dir
    worker_machine = tmp_path / "worker-machine"
    real_ensure = custom_models.ensure_local

    def on_the_worker(row, token=None):
        monkeypatch.setattr(get_settings(), "temp_dir", worker_machine)  # a different machine, a different disk
        return real_ensure(row, token)

    monkeypatch.setattr(custom_models, "ensure_local", on_the_worker)
    monkeypatch.setattr(custom_models, "_fetch_upload", lambda row, dest: (dest / "config.json").write_text("{}"))
    rid = await env.queue_run(custom_model_id=model.id, config={"custom": {"kind": "hf_video", "path": server_path}})

    assert await env.run_worker() == 0
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert run.status == "succeeded" and run.worker_id is not None
    seen = FakeBackend.custom_seen
    assert seen["path"] != server_path and str(worker_machine) in seen["path"]  # the config was re-pointed
    assert seen["has_files"] is True  # and the files were there when training started


async def test_a_model_that_cannot_be_fetched_fails_the_run_with_the_reason(db, env, monkeypatch):
    model = await make_custom_model(db)

    def corrupted(row, dest):
        raise custom_models.CustomModelFileError("The stored bundle no longer matches the checksum")

    monkeypatch.setattr(custom_models, "_fetch_upload", corrupted)
    rid = await env.queue_run(custom_model_id=model.id, config={"custom": {"path": "/server/path"}})

    assert await env.run_worker() == 0
    await get_event_writer().flush()

    run = await run_row(db, rid)
    assert run.status == "failed" and "no longer matches the checksum" in run.failed_message
    assert FakeBackend.custom_seen is None  # training never started
