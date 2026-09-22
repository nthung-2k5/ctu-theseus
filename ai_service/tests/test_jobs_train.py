"""The generic training job engine: run lifecycle, progress, abort, evaluation, log upload —
everything in jobs/train.py that has nothing to do with which backend is training.

Exercised against a FAKE trainer backend (registered for this module only) rather than Ludwig, so
these tests fail only when the generic engine breaks. Ludwig's own glue (the progress callback,
_extract_metrics, LudwigLoadedModel construction) is tested in test_ludwig_train.py.
"""

import asyncio
import logging
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
import sqlalchemy as sa

from theseus.backends.base import EvalResult, LoadedModel, OutputSpec, TrainContext, TrainerBackend
from theseus.backends.base import _registry as backend_registry
from theseus.db.models import RunEvaluation, RunEvent, TrainingMetric, TrainingRun
from theseus.events import InProcessRunEventBus, set_event_writer, set_log_handler
from theseus.events.log_handler import current_run_id, install_run_log_handler, uninstall_run_log_handler
from theseus.events.writer import EventWriter
from theseus.jobs import abort
from theseus.jobs import train as train_job
from theseus.services import storage
from theseus.settings import get_settings

FAKE_LOG_NAMESPACE = "theseus.tests.fake_backend"
fake_logger = logging.getLogger(FAKE_LOG_NAMESPACE)


class FakeLoadedModel(LoadedModel):
    input_columns = ["x"]
    output = OutputSpec(name="class", kind="classification")

    def predict(self, frame):
        return frame

    def to_output(self, predictions, *, top_k=100, input_tokens=None):
        return {"kind": "classification", "feature": "class", "classes": []}

    def golden_prediction(self, predictions, threshold=0.0):
        return {}


class FakeBackend(TrainerBackend):
    """Registers itself at import time like any real backend (see the module-scoped unregister
    fixture below). Its `train()` runs a real epoch loop through the real TrainContext
    (report/check_abort/heartbeat), on the training thread, mirroring what a real backend's
    train() does without needing torch or Ludwig."""

    id = "fake"
    label = "Fake"

    instances: list[SimpleNamespace] = []
    epochs = 3
    losses = [0.9, 0.4, 0.6]
    fail_with: Exception | None = None
    on_epoch_done = None  # callback(epoch), lets a test act between epochs
    eval_result: EvalResult | None = None

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
        instance = SimpleNamespace(seen_run_id=current_run_id.get(), trained_epochs=0)
        cls.instances.append(instance)
        for epoch in range(1, cls.epochs + 1):
            fake_logger.info("Starting epoch %d", epoch)
            run.check_abort()
            run.heartbeat()
            if cls.fail_with:
                raise cls.fail_with
            run.report(epoch, "train", {"loss": cls.losses[epoch - 1] + 0.1, "class.accuracy": 0.5})
            run.report(epoch, "validation", {"loss": cls.losses[epoch - 1]})
            instance.trained_epochs = epoch
            if cls.on_epoch_done:
                cls.on_epoch_done(epoch)
        return FakeLoadedModel()

    @classmethod
    def evaluate(cls, model, df, split_column, item_id_column):
        return cls.eval_result


REPORT = {"split": "test", "outputType": "category", "overall": {"accuracy": 0.9, "macroF1": 0.85}, "classes": ["a"]}


class FakeFrame:
    def __init__(self):
        self.written_to = None

    def to_parquet(self, path):
        self.written_to = path


# FakeBackend registers itself at class-definition time (module import) via __init_subclass__,
# like every real backend — which, for a class defined in a test module, means at pytest
# COLLECTION time, before any test in the whole session runs (collection imports every test
# module up front). Left registered, it would leak into any other file's `list_backends()` /
# `default_backend()` for as long as this module's tests haven't finished executing, and — worse —
# collection order is not guaranteed to match execution order (an explicit file list, `-k`, an
# IDE running one file, a random-order plugin...), so that leak window isn't reliably bounded to
# "before this file's own tests run". Undo the auto-registration immediately, and have the
# function-scoped `env` fixture register/unregister it around each test that actually needs it.
backend_registry.unregister("fake")


@pytest.fixture
async def env(db, monkeypatch, tmp_path, make_run):
    backend_registry.register(FakeBackend)
    FakeBackend.instances = []
    FakeBackend.epochs, FakeBackend.losses = 3, [0.9, 0.4, 0.6]
    FakeBackend.fail_with, FakeBackend.on_epoch_done = None, None
    frame = FakeFrame()
    FakeBackend.eval_result = EvalResult(report=REPORT, predictions=frame)

    bus = InProcessRunEventBus()
    writer = EventWriter(bus)
    await writer.start()
    set_event_writer(writer)
    handler = install_run_log_handler(writer, (FAKE_LOG_NAMESPACE,))
    set_log_handler(handler)

    uploads: dict[str, str] = {}
    json_uploads: dict[str, object] = {}
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    monkeypatch.setattr(train_job.pd, "read_parquet", lambda path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: True)
    monkeypatch.setattr(storage, "delete_prefix", lambda bucket, prefix: 0)
    monkeypatch.setattr(storage, "upload_file", lambda bucket, key, path: uploads.__setitem__(key, open(path).read()))
    monkeypatch.setattr(storage, "upload_json", lambda bucket, key, data: json_uploads.__setitem__(key, data))

    async def _make_run(**kw):
        return await make_run(backend="fake", **kw)

    yield SimpleNamespace(writer=writer, uploads=uploads, json_uploads=json_uploads, make_run=_make_run, frame=frame)

    await writer.stop()
    uninstall_run_log_handler(handler)
    set_log_handler(None)
    set_event_writer(None)
    backend_registry.unregister("fake")


async def run_row(db, rid):
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == rid))).scalar_one()


async def events(db, rid, kind=None):
    async with db() as s:
        q = sa.select(RunEvent.kind, RunEvent.payload).where(RunEvent.run_id == rid).order_by(RunEvent.seq)
        if kind:
            q = q.where(RunEvent.kind == kind)
        return [(k, p) for k, p in (await s.execute(q)).all()]


async def test_a_successful_run_reports_progress_evaluates_and_uploads_its_log(db, env):
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()

    run = await run_row(db, rid)
    assert run.status == "succeeded" and run.best_epoch == 2  # lowest validation loss
    statuses = [p["status"] for _, p in await events(db, rid, "status")]
    assert statuses == ["running", "succeeded"]

    async with db() as s:
        n_metrics = (await s.execute(sa.select(sa.func.count()).select_from(TrainingMetric))).scalar_one()
        evaluation = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert n_metrics == 3 * 3  # per epoch: train loss + class.accuracy + validation loss
    assert (evaluation.status, evaluation.split) == ("success", "test")
    assert evaluation.accuracy == pytest.approx(0.9) and evaluation.macro_f1 == pytest.approx(0.85)  # REAL columns
    assert evaluation.report_key == f"{rid}/evaluation/report.json"
    assert env.json_uploads[f"{rid}/evaluation/report.json"] == REPORT
    assert env.frame.written_to.endswith(f"{rid}/evaluation/predictions.parquet")


async def test_the_run_context_reaches_the_training_thread_and_streams_its_logs(db, env):
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()

    assert FakeBackend.instances[0].seen_run_id == str(rid)  # copied into the training thread
    lines = [p["line"] for _, p in await events(db, rid, "log")]
    assert sum("Starting epoch" in ln for ln in lines) == 3  # logged from the backend's namespace
    log_key = f"{rid}/logs/train.log"
    assert log_key in env.uploads and env.uploads[log_key].count("Starting epoch") == 3
    assert not (get_settings().temp_dir / "logs" / f"{rid}.log").exists()  # local copy removed
    assert current_run_id.get() is None  # the context var does not leak out of the job


async def test_a_run_never_leaks_its_abort_registration_or_log_file_handle(db, env):
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    assert not abort.is_registered(str(rid))
    from theseus.events import get_log_handler

    assert str(rid) not in get_log_handler()._files


async def test_a_failing_evaluation_never_fails_a_run_that_trained_fine(db, env, monkeypatch):
    def boom(cls, model, df, split_column, item_id_column):
        raise RuntimeError("confusion matrix bug")

    monkeypatch.setattr(FakeBackend, "evaluate", classmethod(boom))
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()

    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        ev = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert ev.status == "failed" and "confusion matrix bug" in ev.failed_message


async def test_nothing_to_evaluate_skips_the_report_but_the_run_still_succeeds(db, env):
    FakeBackend.eval_result = None
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()
    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0


async def test_a_training_error_fails_the_run_once_with_its_message_and_still_uploads_the_log(db, env):
    FakeBackend.fail_with = RuntimeError("CUDA out of memory")
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)  # must not raise
    await env.writer.flush()

    run = await run_row(db, rid)
    assert run.status == "failed" and run.failed_message == "RuntimeError: CUDA out of memory"
    assert [p["status"] for _, p in await events(db, rid, "status")] == ["running", "failed"]
    assert f"{rid}/logs/train.log" in env.uploads
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0


async def test_a_missing_config_fails_the_run_before_any_model_is_built(db, env, monkeypatch):
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: False)
    rid = await env.make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()
    run = await run_row(db, rid)
    assert run.status == "failed" and "No training config" in run.failed_message
    assert FakeBackend.instances == []


async def test_a_run_canceled_before_it_starts_never_builds_a_model(db, env):
    rid = await env.make_run(status="running", cancel_requested_at=sa.func.now())
    await train_job.run_train(rid)
    await env.writer.flush()
    assert (await run_row(db, rid)).status == "canceled"
    assert FakeBackend.instances == []


async def test_cancel_mid_training_stops_at_the_next_epoch_boundary_and_ends_canceled(db, env):
    """The real cancel path: the API sets the flag while a worker thread trains; check_abort raises."""
    FakeBackend.epochs, FakeBackend.losses = 6, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    epoch_one_done, resume = threading.Event(), threading.Event()

    def on_epoch_done(epoch):
        if epoch == 1:
            epoch_one_done.set()
            resume.wait(5)  # hold the thread until the cancel request has landed

    FakeBackend.on_epoch_done = on_epoch_done
    rid = await env.make_run(status="running")
    job = asyncio.create_task(train_job.run_train(rid))
    await asyncio.to_thread(epoch_one_done.wait, 5)

    assert await abort.request_cancel(rid) is True
    resume.set()
    await asyncio.wait_for(job, 10)
    await env.writer.flush()

    instance = FakeBackend.instances[0]
    assert instance.trained_epochs == 1  # aborted at the boundary right after the cancel, not run to completion
    run = await run_row(db, rid)
    assert run.status == "canceled" and run.cancel_requested_at is not None
    assert [p["status"] for _, p in await events(db, rid, "status")] == ["running", "canceled"]  # exactly one terminal
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0
    assert f"{rid}/logs/train.log" in env.uploads
