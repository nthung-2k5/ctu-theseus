"""The real lifespan wired end to end: writer, log handler, recovery, dispatcher lanes, reaper."""

import asyncio
import logging
import sys

import pytest
import sqlalchemy as sa

from theseus import events as events_pkg
from theseus.app import create_app
from theseus.db.models import TrainingRun
from theseus.events import get_event_bus
from theseus.events.stream import stream_run_events
from theseus.jobs import abort
from theseus.jobs.dispatcher import get_dispatcher, nudge
from theseus.services import storage


@pytest.fixture(autouse=True)
def no_s3(monkeypatch):
    monkeypatch.setattr(storage, "ensure_buckets", lambda: None)
    monkeypatch.setattr(
        storage, "file_exists", lambda bucket, key: False
    )  # so a training run fails fast, not on Ludwig
    monkeypatch.setattr(storage, "upload_file", lambda *a, **k: None)


async def run_status(db, rid):
    async with db() as s:
        return (
            await s.execute(sa.select(TrainingRun.status, TrainingRun.failed_message).where(TrainingRun.id == rid))
        ).one()


async def wait_for_status(db, rid, want, timeout=20):
    async def poll():
        while (await run_status(db, rid))[0] != want:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout)


async def test_startup_and_shutdown_install_and_remove_every_process_wide_component(db):
    app = create_app()
    async with app.router.lifespan_context(app):
        assert events_pkg.get_event_writer() is not None
        assert events_pkg.get_log_handler() in logging.getLogger("ludwig").handlers
        d = get_dispatcher()
        assert d is not None and set(d.lanes) == {"train", "export", "inference"}
        assert [d.lanes[n].concurrency for n in ("train", "export")] == [1, 2]

    assert get_dispatcher() is None
    assert not any(type(h).__name__ == "RunLogHandler" for h in logging.getLogger("ludwig").handlers)
    with pytest.raises(RuntimeError, match="not running"):
        events_pkg.get_event_writer()


async def test_a_queued_run_is_claimed_run_and_its_events_are_streamable_end_to_end(db, make_run):
    rid = await make_run(status="queued")  # its config does not exist (file_exists is stubbed False), so it fails fast
    app = create_app()
    async with app.router.lifespan_context(app):
        nudge("train")
        await wait_for_status(db, rid, "failed")
        await events_pkg.get_event_writer().flush()

        status, message = await run_status(db, rid)
        assert status == "failed" and "No training config" in message

        got = []
        async for ev in stream_run_events(get_event_bus(), rid, after_seq=0, keepalive_seconds=0.2):
            if ev is None:
                break
            got.append(ev)
        statuses = [e["payload"]["status"] for e in got if e["kind"] == "status"]
        assert statuses == ["running", "failed"]
        assert any(e["kind"] == "log" for e in got)  # the job own log lines were streamed too
        assert [e["seq"] for e in got] == sorted(e["seq"] for e in got)
    assert not abort.is_registered(str(rid))


async def test_startup_recovery_runs_before_any_job_is_claimed(db, make_run):
    stale = await make_run(status="running")  # left over from a process that died mid-training
    app = create_app()
    async with app.router.lifespan_context(app):
        await events_pkg.get_event_writer().flush()
        status, message = await run_status(db, stale)
        assert status == "failed" and message == "Server restarted during training"  # never resumed


async def test_the_service_refuses_to_start_with_more_than_one_worker(db, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["uvicorn", "--workers", "4"])
    app = create_app()
    with pytest.raises(RuntimeError, match="single process"):
        async with app.router.lifespan_context(app):
            pass
