"""Abort, startup recovery, reapers and the dispatcher: how jobs live, die and get recovered."""

import asyncio
import uuid
from datetime import timedelta

import pytest
import sqlalchemy as sa

from theseus.db.models import DatasetVersion, InferenceJob, ModelExport, RunEvent, TrainingRun
from theseus.events import InProcessRunEventBus, set_event_writer
from theseus.events.writer import EventWriter
from theseus.jobs import abort, queue, reapers
from theseus.jobs import inference as inference_jobs
from theseus.jobs.dispatcher import Dispatcher, Lane
from theseus.jobs.recovery import recover_on_startup
from theseus.services import storage


@pytest.fixture
async def writer(db):
    w = EventWriter(InProcessRunEventBus())
    await w.start()
    set_event_writer(w)
    yield w
    await w.stop()
    set_event_writer(None)


@pytest.fixture
def deleted_keys(monkeypatch):
    keys: list[str] = []
    monkeypatch.setattr(storage, "delete_file", lambda bucket, key: keys.append(key))
    return keys


async def one(db, model, id_):
    async with db() as s:
        return (await s.execute(sa.select(model).where(model.id == id_))).scalar_one()


async def status_events(db, run_id):
    async with db() as s:
        rows = (
            await s.execute(
                sa.select(RunEvent.payload)
                .where(RunEvent.run_id == run_id, RunEvent.kind == "status")
                .order_by(RunEvent.seq)
            )
        ).all()
    return [(r[0]["status"], r[0].get("message")) for r in rows]


# -- Abort -----------------------------------------------------------------------------------


def test_training_aborted_is_a_plain_exception_never_a_keyboard_interrupt():
    assert issubclass(abort.TrainingAborted, Exception)
    assert not issubclass(abort.TrainingAborted, (KeyboardInterrupt, SystemExit))
    assert not issubclass(abort.TrainingAborted, BaseException) or issubclass(abort.TrainingAborted, Exception)


async def test_cancel_marks_intent_durably_signals_the_thread_and_ends_the_run_immediately(db, writer, make_run):
    rid = await make_run(status="running")
    event = abort.register(str(rid))
    try:
        assert await abort.request_cancel(rid) is True
        await writer.flush()
        assert event.is_set()  # the training thread will see it at its next epoch boundary
        run = await one(db, TrainingRun, rid)
        assert run.status == "canceled" and run.cancel_requested_at is not None
        assert [s for s, _ in await status_events(db, rid)] == ["canceled"]
    finally:
        abort.unregister(str(rid))


async def test_cancelling_a_queued_run_means_it_is_never_claimed(db, writer, make_run):
    rid = await make_run(status="queued")
    assert await abort.request_cancel(rid) is True
    await writer.flush()
    assert await queue.claim_one(queue.TRAIN, "w", 300) is None


async def test_cancelling_a_finished_run_is_refused_and_emits_nothing(db, writer, make_run):
    rid = await make_run(status="succeeded")
    assert await abort.request_cancel(rid) is False
    await writer.flush()
    assert await status_events(db, rid) == []
    assert (await one(db, TrainingRun, rid)).status == "succeeded"


async def test_a_thread_finishing_after_cancel_cannot_overwrite_the_canceled_state(db, writer, make_run):
    rid = await make_run(status="running")
    await abort.request_cancel(rid)
    writer.status(rid, "succeeded")  # what the still-running training thread would eventually emit
    await writer.flush()
    assert (await one(db, TrainingRun, rid)).status == "canceled"
    assert [s for s, _ in await status_events(db, rid)] == ["canceled"]


# -- Startup recovery ------------------------------------------------------------------------


async def test_recovery_fails_running_runs_and_never_resumes_them_but_leaves_queued_ones(db, writer, make_run):
    running = await make_run(status="running")
    queued = await make_run(status="queued")
    finished = await make_run(status="succeeded")

    report = await recover_on_startup()

    assert report.runs_failed == 1
    r = await one(db, TrainingRun, running)
    assert r.status == "failed" and r.failed_message == "Server restarted during training"
    assert await status_events(db, running) == [("failed", "Server restarted during training")]
    assert (await one(db, TrainingRun, queued)).status == "queued"
    assert (await one(db, TrainingRun, finished)).status == "succeeded"


async def test_recovery_requeues_exports_with_attempts_left_and_fails_the_rest(db, writer, make_export):
    retry, _ = await make_export(status="assembling", attempt=1)
    exhausted, _ = await make_export(status="converting", attempt=3)
    untouched, _ = await make_export(status="ready")

    report = await recover_on_startup()

    assert (report.exports_requeued, report.exports_failed) == (1, 1)
    assert (await one(db, ModelExport, retry)).status == "pending"
    e = await one(db, ModelExport, exhausted)
    assert e.status == "failed" and "restarted" in e.failed_message
    assert (await one(db, ModelExport, untouched)).status == "ready"


async def test_recovery_only_requeues_inference_whose_input_survived(db, writer, make_inference, deleted_keys):
    text, _ = await make_inference(status="running", attempt=1)
    durable, _ = await make_inference(
        status="running", attempt=1, payload={"kind": "file", "filename": "a.png"}, upload_key="inference/x/input.png"
    )
    lost_temp, _ = await make_inference(
        status="running", attempt=1, payload={"kind": "file", "filename": "b.png", "localPath": "/nonexistent/b.png"}
    )
    exhausted, _ = await make_inference(status="running", attempt=3)

    report = await recover_on_startup()

    assert (report.inference_requeued, report.inference_failed) == (2, 2)
    assert (await one(db, InferenceJob, text)).status == "pending"
    assert (await one(db, InferenceJob, durable)).status == "pending"
    for dead in (lost_temp, exhausted):
        j = await one(db, InferenceJob, dead)
        assert j.status == "failed" and j.completed_at is not None
        assert j.error == inference_jobs.GENERIC_FAILURE  # the client never sees internals


async def test_recovery_fails_snapshots_left_building(db, writer, make_run):
    rid = await make_run()
    async with db() as s:
        version_id = (
            await s.execute(sa.select(TrainingRun.dataset_version_id).where(TrainingRun.id == rid))
        ).scalar_one()
        await s.execute(sa.update(DatasetVersion).where(DatasetVersion.id == version_id).values(status="building"))
        await s.commit()
    assert (await recover_on_startup()).snapshots_failed == 1
    assert (await one(db, DatasetVersion, version_id)).status == "failed"


# -- Reapers ---------------------------------------------------------------------------------


async def test_stale_run_reaper_fails_only_running_runs_that_have_gone_silent(db, writer, make_run):
    silent = await make_run(status="running")
    fresh = await make_run(status="running")
    queued = await make_run(status="queued")
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == silent)
            .values(heartbeat_at=sa.func.now() - timedelta(hours=2))
        )
        await s.execute(sa.update(TrainingRun).where(TrainingRun.id == fresh).values(heartbeat_at=sa.func.now()))
        await s.execute(  # a queued run waiting behind another has no heartbeat and is not stale
            sa.update(TrainingRun).where(TrainingRun.id == queued).values(created_at=sa.func.now() - timedelta(days=1))
        )
        await s.commit()

    assert await reapers.stale_runs(timeout_seconds=900) == 1
    await writer.flush()
    assert (await one(db, TrainingRun, silent)).status == "failed"
    assert (await one(db, TrainingRun, fresh)).status == "running"
    assert (await one(db, TrainingRun, queued)).status == "queued"


async def test_stale_pending_inference_is_failed_and_its_upload_deleted(db, make_inference, deleted_keys):
    old, _ = await make_inference(upload_key="inference/old/input.png", payload={"kind": "file", "filename": "x"})
    recent, _ = await make_inference()
    async with db() as s:
        await s.execute(
            sa.update(InferenceJob).where(InferenceJob.id == old).values(created_at=sa.func.now() - timedelta(hours=2))
        )
        await s.commit()

    assert await reapers.stale_pending_inference() == 1
    assert (await one(db, InferenceJob, old)).status == "failed"
    assert (await one(db, InferenceJob, recent)).status == "pending"
    assert deleted_keys == ["inference/old/input.png"]
    assert (await one(db, InferenceJob, old)).upload_key is None


async def test_terminal_uploads_are_reaped_by_row_state_never_by_age(db, make_inference, deleted_keys):
    done, _ = await make_inference(status="success", upload_key="inference/done/input.png", payload={"kind": "file"})
    running, _ = await make_inference(
        status="running", upload_key="inference/live/input.csv", payload={"kind": "batch"}, attempt=1
    )
    async with db() as s:  # the in-flight batch job is old, but its input must survive
        await s.execute(
            sa.update(InferenceJob)
            .where(InferenceJob.id == running)
            .values(created_at=sa.func.now() - timedelta(days=3))
        )
        await s.commit()

    assert await reapers.terminal_uploads() == 1
    assert deleted_keys == ["inference/done/input.png"]
    assert (await one(db, InferenceJob, running)).upload_key == "inference/live/input.csv"


async def test_event_retention_drops_old_logs_of_finished_runs_and_very_old_events(db, writer, make_run):
    rid = await make_run(status="succeeded")
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun).where(TrainingRun.id == rid).values(completed_at=sa.func.now() - timedelta(days=2))
        )
        for kind, age_days in (("log", 0), ("status", 0), ("metric", 8)):
            await s.execute(
                sa.insert(RunEvent).values(
                    run_id=rid, kind=kind, ts=sa.func.now() - timedelta(days=age_days), payload={"kind": kind}
                )
            )
        await s.commit()

    assert await reapers.event_retention() == 2  # the log (run finished >24h ago) and the 8-day-old metric
    async with db() as s:
        kinds = (await s.execute(sa.select(RunEvent.kind))).scalars().all()
    assert kinds == ["status"]


async def test_one_failing_reaper_does_not_stop_the_others(db, monkeypatch):
    calls: list[str] = []

    async def boom():
        raise RuntimeError("bad")

    async def fine():
        calls.append("fine")
        return 1

    boom.__name__, fine.__name__ = "boom", "fine"
    monkeypatch.setattr(reapers, "TASKS", (boom, fine))
    assert await reapers.run_once() == {"fine": 1}
    assert calls == ["fine"]


# -- Dispatcher ------------------------------------------------------------------------------


class Probe:
    """Records how many jobs of a lane ran at once, and finishes them by marking the run succeeded."""

    def __init__(self, db, hold: float = 0.05):
        self.db, self.hold = db, hold
        self.active = self.peak = 0
        self.done: list[uuid.UUID] = []

    async def run(self, job_id: uuid.UUID) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.hold)
        async with self.db() as s:
            await s.execute(sa.update(TrainingRun).where(TrainingRun.id == job_id).values(status="succeeded"))
            await s.commit()
        self.active -= 1
        self.done.append(job_id)


async def wait_until(predicate, timeout=8.0):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


async def test_train_lane_never_runs_two_jobs_at_once_yet_drains_the_whole_queue(db, make_run):
    ids = [await make_run() for _ in range(4)]
    probe = Probe(db)
    d = Dispatcher([Lane("train", queue.TRAIN, 1, probe.run)], poll_interval=0.05)
    await d.start()
    try:
        await wait_until(lambda: len(probe.done) == 4)
    finally:
        await d.stop()
    assert probe.peak == 1
    assert probe.done == ids  # FIFO: a sweep of trials trains in the order it was queued


async def test_a_lane_with_concurrency_two_runs_two_at_once_not_more(db, make_run):
    for _ in range(6):
        await make_run()
    probe = Probe(db, hold=0.15)
    d = Dispatcher([Lane("train", queue.TRAIN, 2, probe.run)], poll_interval=0.05)
    await d.start()
    try:
        await wait_until(lambda: len(probe.done) == 6)
    finally:
        await d.stop()
    assert probe.peak == 2


async def test_nudge_picks_up_new_work_immediately_instead_of_waiting_for_the_poll(db, make_run):
    probe = Probe(db, hold=0)
    d = Dispatcher([Lane("train", queue.TRAIN, 1, probe.run)], poll_interval=30)  # poll would be far too slow
    await d.start()
    try:
        await asyncio.sleep(0.2)  # the lane is now idle, waiting on its wakeup
        rid = await make_run()
        d.nudge("train")
        await wait_until(lambda: probe.done == [rid], timeout=3)
    finally:
        await d.stop()


async def test_a_failing_job_invokes_its_failure_handler_and_frees_the_slot(db, make_run):
    first, second = await make_run(), await make_run()
    failures: list[uuid.UUID] = []
    ran: list[uuid.UUID] = []

    async def run(job_id):
        if job_id == first:
            raise RuntimeError("kaboom")
        ran.append(job_id)

    async def on_failure(job_id, exc):
        failures.append(job_id)

    d = Dispatcher([Lane("train", queue.TRAIN, 1, run, on_failure)], poll_interval=0.05)
    await d.start()
    try:
        await wait_until(lambda: ran == [second])
    finally:
        await d.stop()
    assert failures == [first]  # the second job still ran: one failure never wedges the lane


async def test_leases_are_renewed_while_a_long_job_runs(db, make_inference):
    job_id, _ = await make_inference()
    seen: list = []

    async def run(jid):
        for _ in range(4):
            await asyncio.sleep(0.2)
            async with db() as s:
                seen.append(
                    (
                        await s.execute(sa.select(InferenceJob.lease_expires_at).where(InferenceJob.id == jid))
                    ).scalar_one()
                )
        await asyncio.sleep(0)

    d = Dispatcher(
        [Lane("inference", queue.INFERENCE, 1, run, renew_lease=True)],
        poll_interval=0.05,
        lease_seconds=60,
        renew_interval=0.15,
    )
    await d.start()
    try:
        await wait_until(lambda: len(seen) == 4)
    finally:
        await d.stop()
    assert seen[-1] > seen[0]  # the lease moved forward while the job was still running


async def test_stopping_the_dispatcher_signals_training_threads_and_cancels_stragglers(db, make_run):
    rid = await make_run()
    started = asyncio.Event()

    async def run(job_id):
        abort.register(str(job_id))
        started.set()
        await asyncio.sleep(60)

    d = Dispatcher([Lane("train", queue.TRAIN, 1, run)], poll_interval=0.05)
    await d.start()
    await asyncio.wait_for(started.wait(), 3)
    ev = abort._events[str(rid)]
    await d.stop(timeout=0.2)
    assert ev.is_set()  # the thread would stop at its next epoch boundary
    assert d.running_count("train") == 0
    abort.unregister(str(rid))
