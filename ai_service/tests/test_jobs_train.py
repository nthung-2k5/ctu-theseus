import asyncio
import logging
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
import sqlalchemy as sa

from theseus.db.models import RunEvaluation, RunEvent, TrainingMetric, TrainingRun
from theseus.events import InProcessRunEventBus, set_event_writer, set_log_handler
from theseus.events.log_handler import current_run_id, install_run_log_handler, uninstall_run_log_handler
from theseus.events.writer import EventWriter
from theseus.jobs import abort
from theseus.jobs import train as train_job
from theseus.services import storage
from theseus.settings import get_settings


class Metric:
    def __init__(self, value):
        self.value = value


def tracker(epoch, val_loss):
    return SimpleNamespace(
        epoch=epoch,
        train_metrics={"combined": {"loss": [Metric(val_loss + 0.1)]}, "class": {"accuracy": [Metric(0.5)]}},
        validation_metrics={"combined": {"loss": [Metric(val_loss)]}},
        test_metrics={},
    )


class FakeModel:
    """Stands in for LudwigModel: runs epochs through the REAL callbacks, on the training thread."""

    instances: list["FakeModel"] = []
    epochs = 3
    losses = [0.9, 0.4, 0.6]
    fail_with: Exception | None = None
    on_epoch_done = None  # callback(epoch), lets a test act between epochs

    def __init__(self, config, logging_level=None, callbacks=None):
        self.config, self.callbacks = config, callbacks or []
        self.trained_epochs = 0
        self.seen_run_id = None
        FakeModel.instances.append(self)

    def train(self, dataset, output_directory, experiment_name):
        self.seen_run_id = current_run_id.get()  # proves the contextvar reached the training thread
        for epoch in range(1, self.epochs + 1):
            logging.getLogger("ludwig.trainers.trainer").info("Starting epoch %d", epoch)
            for cb in self.callbacks:
                cb.on_epoch_start(None, tracker(epoch, self.losses[epoch - 1]), None)
            if self.fail_with:
                raise self.fail_with
            for cb in self.callbacks:
                cb.on_epoch_end(None, tracker(epoch, self.losses[epoch - 1]), None)
            self.trained_epochs = epoch
            if FakeModel.on_epoch_done:
                FakeModel.on_epoch_done(epoch)
        return "trained"


class FakeFrame:
    def __init__(self):
        self.written_to = None

    def to_parquet(self, path):
        self.written_to = path


REPORT = {"split": "test", "outputType": "category", "overall": {"accuracy": 0.9, "macroF1": 0.85}, "classes": ["a"]}


@pytest.fixture
async def env(db, monkeypatch, tmp_path):
    FakeModel.instances = []
    FakeModel.epochs, FakeModel.losses, FakeModel.fail_with, FakeModel.on_epoch_done = 3, [0.9, 0.4, 0.6], None, None

    bus = InProcessRunEventBus()
    writer = EventWriter(bus)
    await writer.start()
    set_event_writer(writer)
    handler = install_run_log_handler(writer)
    set_log_handler(handler)

    uploads: dict[str, str] = {}
    json_uploads: dict[str, object] = {}
    frame = FakeFrame()
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    monkeypatch.setattr(train_job, "LudwigModel", FakeModel)
    monkeypatch.setattr(train_job.pd, "read_parquet", lambda path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(train_job, "build_evaluation_report", lambda *a, **k: (REPORT, frame))
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: True)
    monkeypatch.setattr(storage, "delete_prefix", lambda bucket, prefix: 0)
    monkeypatch.setattr(storage, "upload_file", lambda bucket, key, path: uploads.__setitem__(key, open(path).read()))
    monkeypatch.setattr(storage, "upload_json", lambda bucket, key, data: json_uploads.__setitem__(key, data))

    yield SimpleNamespace(writer=writer, uploads=uploads, json_uploads=json_uploads, frame=frame)

    await writer.stop()
    uninstall_run_log_handler(handler)
    set_log_handler(None)
    set_event_writer(None)


async def run_row(db, rid):
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == rid))).scalar_one()


async def events(db, rid, kind=None):
    async with db() as s:
        q = sa.select(RunEvent.kind, RunEvent.payload).where(RunEvent.run_id == rid).order_by(RunEvent.seq)
        if kind:
            q = q.where(RunEvent.kind == kind)
        return [(k, p) for k, p in (await s.execute(q)).all()]


async def test_a_successful_run_reports_progress_evaluates_and_uploads_its_log(db, env, make_run):
    rid = await make_run(status="running")
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


async def test_per_feature_metrics_are_namespaced_and_combined_is_not(db, env, make_run):
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()
    (train_event,) = [p for _, p in await events(db, rid, "metric") if p["epoch"] == 1 and p["split"] == "train"]
    assert train_event["metrics"] == {"loss": pytest.approx(1.0), "class.accuracy": 0.5}


async def test_the_run_context_reaches_the_training_thread_and_streams_its_logs(db, env, make_run):
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()

    assert FakeModel.instances[0].seen_run_id == str(rid)  # copied into the training thread
    lines = [p["line"] for _, p in await events(db, rid, "log")]
    assert sum("Starting epoch" in ln for ln in lines) == 3  # logged from the ludwig namespace in the thread
    log_key = f"{rid}/logs/train.log"
    assert log_key in env.uploads and env.uploads[log_key].count("Starting epoch") == 3
    assert not (get_settings().temp_dir / "logs" / f"{rid}.log").exists()  # local copy removed
    assert current_run_id.get() is None  # the context var does not leak out of the job


async def test_a_run_never_leaks_its_abort_registration_or_log_file_handle(db, env, make_run):
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    assert not abort.is_registered(str(rid))
    from theseus.events import get_log_handler

    assert str(rid) not in get_log_handler()._files


async def test_a_failing_evaluation_never_fails_a_run_that_trained_fine(db, env, make_run, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("confusion matrix bug")

    monkeypatch.setattr(train_job, "build_evaluation_report", boom)
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()

    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        ev = (await s.execute(sa.select(RunEvaluation))).scalar_one()
    assert ev.status == "failed" and "confusion matrix bug" in ev.failed_message


async def test_nothing_to_evaluate_skips_the_report_but_the_run_still_succeeds(db, env, make_run, monkeypatch):
    monkeypatch.setattr(train_job, "build_evaluation_report", lambda *a, **k: None)
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()
    assert (await run_row(db, rid)).status == "succeeded"
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0


async def test_a_training_error_fails_the_run_once_with_its_message_and_still_uploads_the_log(db, env, make_run):
    FakeModel.fail_with = RuntimeError("CUDA out of memory")
    rid = await make_run(status="running")
    await train_job.run_train(rid)  # must not raise
    await env.writer.flush()

    run = await run_row(db, rid)
    assert run.status == "failed" and run.failed_message == "RuntimeError: CUDA out of memory"
    assert [p["status"] for _, p in await events(db, rid, "status")] == ["running", "failed"]
    assert f"{rid}/logs/train.log" in env.uploads
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0


async def test_a_missing_config_fails_the_run_before_any_model_is_built(db, env, make_run, monkeypatch):
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: False)
    rid = await make_run(status="running")
    await train_job.run_train(rid)
    await env.writer.flush()
    run = await run_row(db, rid)
    assert run.status == "failed" and "No training config" in run.failed_message
    assert FakeModel.instances == []


async def test_a_run_canceled_before_it_starts_never_builds_a_model(db, env, make_run):
    rid = await make_run(status="running", cancel_requested_at=sa.func.now())
    await train_job.run_train(rid)
    await env.writer.flush()
    assert (await run_row(db, rid)).status == "canceled"
    assert FakeModel.instances == []


async def test_cancel_mid_training_stops_at_the_next_epoch_boundary_and_ends_canceled(db, env, make_run):
    """The real cancel path: the API sets the flag while a worker thread trains; the callback raises."""
    FakeModel.epochs, FakeModel.losses = 6, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    epoch_one_done, resume = threading.Event(), threading.Event()

    def on_epoch_done(epoch):
        if epoch == 1:
            epoch_one_done.set()
            resume.wait(5)  # hold the thread until the cancel request has landed

    FakeModel.on_epoch_done = on_epoch_done
    rid = await make_run(status="running")
    job = asyncio.create_task(train_job.run_train(rid))
    await asyncio.to_thread(epoch_one_done.wait, 5)

    assert await abort.request_cancel(rid) is True
    resume.set()
    await asyncio.wait_for(job, 10)
    await env.writer.flush()

    model = FakeModel.instances[0]
    assert model.trained_epochs == 1  # aborted at the boundary right after the cancel, not run to completion
    run = await run_row(db, rid)
    assert run.status == "canceled" and run.cancel_requested_at is not None
    assert [p["status"] for _, p in await events(db, rid, "status")] == ["running", "canceled"]  # exactly one terminal
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvaluation))).scalar_one() == 0
    assert f"{rid}/logs/train.log" in env.uploads


def test_metric_extraction_prefixes_output_features_but_not_the_combined_aggregate():
    out = train_job._extract_metrics(
        {"combined": {"loss": [Metric(0.1), Metric(0.2)]}, "label": {"accuracy": [Metric(0.9)], "empty": []}}
    )
    assert out == {"loss": 0.2, "label.accuracy": 0.9}  # latest value; empty history skipped
