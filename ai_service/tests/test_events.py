import asyncio
import contextvars
import logging
import math
import threading

import pytest
import sqlalchemy as sa

from theseus.db.models import RunEvent, TrainingMetric, TrainingRun
from theseus.events import bus as bus_module
from theseus.events.bus import InProcessRunEventBus
from theseus.events.log_handler import RunLogHandler, current_run_id, install_run_log_handler, uninstall_run_log_handler
from theseus.events.stream import stream_run_events
from theseus.events.writer import EventWriter


@pytest.fixture
async def bus():
    return InProcessRunEventBus()


@pytest.fixture
async def writer(db, bus):
    w = EventWriter(bus, log_cap=5)
    await w.start()
    yield w
    await w.stop()


async def run_row(db, run_id) -> TrainingRun:
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()


async def events(db, run_id, kind=None):
    async with db() as s:
        q = sa.select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
        if kind:
            q = q.where(RunEvent.kind == kind)
        return (await s.execute(q)).scalars().all()


# -- Status projection -----------------------------------------------------------------------


async def test_status_transitions_update_the_run_and_record_events(db, writer, make_run):
    rid = await make_run()
    writer.status(rid, "running")
    writer.status(rid, "succeeded")
    await writer.flush()

    run = await run_row(db, rid)
    assert run.status == "succeeded"
    assert run.started_at is not None and run.completed_at is not None and run.heartbeat_at is not None
    evs = await events(db, rid, "status")
    assert [e.payload["status"] for e in evs] == ["running", "succeeded"]
    assert all(e.payload["runId"] == str(rid) for e in evs)


async def test_a_terminal_run_is_never_resurrected_and_stale_events_are_not_recorded(db, writer, make_run):
    rid = await make_run(status="running")
    writer.status(rid, "canceled")
    await writer.flush()
    canceled_at = (await run_row(db, rid)).completed_at

    # Late events from the training thread arrive after the cancel endpoint already ended the run.
    writer.status(rid, "running")
    writer.metric(rid, 1, "train", {"loss": 0.5})
    writer.status(rid, "succeeded")
    await writer.flush()

    run = await run_row(db, rid)
    assert run.status == "canceled" and run.completed_at == canceled_at
    assert [e.payload["status"] for e in await events(db, rid, "status")] == ["canceled"]
    assert await events(db, rid, "metric") == []


async def test_failed_status_stores_the_message_and_started_at_is_set_only_once(db, writer, make_run):
    rid = await make_run()
    writer.status(rid, "running")
    await writer.flush()
    first = (await run_row(db, rid)).started_at
    await asyncio.sleep(0.02)
    writer.status(rid, "running")
    writer.status(rid, "failed", "CUDA out of memory")
    await writer.flush()

    run = await run_row(db, rid)
    assert run.started_at == first
    assert (run.status, run.failed_message) == ("failed", "CUDA out of memory")


# -- Metrics ---------------------------------------------------------------------------------


async def test_metrics_upsert_and_track_the_lowest_validation_loss_as_best_epoch(db, writer, make_run):
    rid = await make_run(status="running")
    for epoch, loss in [(1, 0.9), (2, 0.5), (3, 0.7)]:
        writer.metric(rid, epoch, "validation", {"loss": loss, "accuracy": 1 - loss})
        writer.metric(rid, epoch, "train", {"loss": loss - 0.1})
    await writer.flush()

    run = await run_row(db, rid)
    assert run.best_epoch == 2  # train loss never counts, only validation
    async with db() as s:
        rows = (await s.execute(sa.select(TrainingMetric).where(TrainingMetric.training_run_id == rid))).scalars().all()
    assert len(rows) == 3 * 2 + 3  # (loss+accuracy) x 3 validation epochs + loss x 3 train epochs


async def test_re_emitting_an_epoch_overwrites_instead_of_duplicating_or_erroring(db, writer, make_run):
    rid = await make_run(status="running")
    writer.metric(rid, 1, "validation", {"loss": 0.9})
    writer.metric(rid, 1, "validation", {"loss": 0.4})
    await writer.flush()
    async with db() as s:
        rows = (
            await s.execute(sa.select(TrainingMetric.metric_value).where(TrainingMetric.training_run_id == rid))
        ).all()
    assert [r[0] for r in rows] == pytest.approx([0.4])


async def test_non_finite_metrics_are_dropped_not_written(db, writer, make_run):
    rid = await make_run(status="running")
    writer.metric(rid, 1, "train", {"loss": math.nan, "accuracy": 0.8, "weird": math.inf})
    await writer.flush()
    (ev,) = await events(db, rid, "metric")
    assert ev.payload["metrics"] == {"accuracy": 0.8}


# -- Logs ------------------------------------------------------------------------------------


async def test_log_stream_is_capped_with_one_truncation_notice_and_heartbeat_is_throttled(db, writer, make_run):
    rid = await make_run(status="running")
    for i in range(9):
        writer.log(rid, "info", f"line {i}")
    await writer.flush()

    logs = await events(db, rid, "log")
    assert len(logs) == 5 + 1  # cap + one notice, the remaining lines are dropped
    assert logs[-1].payload["level"] == "warn" and "truncated" in logs[-1].payload["line"]
    assert [e.payload["line"] for e in logs[:5]] == [f"line {i}" for i in range(5)]
    assert (await run_row(db, rid)).heartbeat_at is not None


# -- Ordering and publishing -----------------------------------------------------------------


async def test_concurrent_emitters_get_unique_gapless_seqs_preserving_per_thread_order(db, writer, make_run):
    rid = await make_run(status="running")

    def emitter(tid: int):
        for n in range(40):
            writer.metric(rid, n, "train", {f"t{tid}": float(n)})

    threads = [threading.Thread(target=emitter, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    await writer.flush()

    evs = await events(db, rid, "metric")
    seqs = [e.seq for e in evs]
    assert len(evs) == 6 * 40
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    for tid in range(6):
        epochs = [e.payload["epoch"] for e in evs if f"t{tid}" in e.payload["metrics"]]
        assert epochs == list(range(40))


async def test_events_are_published_to_the_bus_only_after_commit_with_their_db_seq(db, writer, bus, make_run):
    rid = await make_run(status="running")
    sub = bus.subscribe(str(rid))
    writer.status(rid, "succeeded")
    event = await asyncio.wait_for(sub.queue.get(), 2)

    async with db() as s:
        stored = (await s.execute(sa.select(RunEvent.seq).where(RunEvent.run_id == rid))).scalar_one()
    assert event["seq"] == stored  # already committed by the time a subscriber sees it
    assert event["payload"]["status"] == "succeeded"


async def test_writer_survives_an_event_for_a_missing_run(db, writer, make_run):
    import uuid

    good = await make_run(status="running")
    writer.log(uuid.uuid4(), "info", "orphan line")  # FK violation
    writer.status(good, "succeeded")
    await writer.flush()
    assert (await run_row(db, good)).status == "succeeded"


async def test_emitting_before_start_is_an_error(bus):
    with pytest.raises(RuntimeError, match="not been started"):
        EventWriter(bus).status("00000000-0000-0000-0000-000000000000", "running")


# -- SSE stream ------------------------------------------------------------------------------


async def collect(agen, until, timeout=5):
    out = []

    async def run():
        async for ev in agen:
            if ev is None:
                continue
            out.append(ev)
            if until(ev):
                return

    await asyncio.wait_for(run(), timeout)
    return out


async def test_stream_replays_backlog_after_last_event_id_then_continues_live(db, writer, bus, make_run):
    rid = await make_run(status="running")
    for n in range(1, 6):
        writer.metric(rid, n, "train", {"loss": 1 / n})
    await writer.flush()
    evs = await events(db, rid)
    resume_from = evs[1].seq  # client already has the first two events

    stream = stream_run_events(bus, rid, after_seq=resume_from)
    task = asyncio.create_task(collect(stream, lambda e: e["payload"].get("status") == "succeeded"))
    await asyncio.sleep(0.1)
    writer.metric(rid, 6, "train", {"loss": 0.1})
    writer.status(rid, "succeeded")
    got = await task

    assert [e["seq"] for e in got] == [e.seq for e in evs[2:]] + [got[-2]["seq"], got[-1]["seq"]]
    assert [e["payload"].get("epoch") for e in got[:-1]] == [3, 4, 5, 6]


async def test_stream_has_no_gap_or_duplicate_when_events_arrive_during_the_backlog_read(db, writer, bus, make_run):
    rid = await make_run(status="running")
    total = 300
    stream = stream_run_events(bus, rid, after_seq=0)

    async def produce():
        for n in range(total):
            writer.metric(rid, n, "train", {"loss": float(n)})
            if n % 25 == 0:
                await asyncio.sleep(0)
        writer.status(rid, "succeeded")

    consumer = asyncio.create_task(collect(stream, lambda e: e["payload"].get("status") == "succeeded", timeout=20))
    await asyncio.sleep(0.05)
    await produce()
    got = await consumer

    seqs = [e["seq"] for e in got]
    assert len(seqs) == total + 1
    assert seqs == sorted(set(seqs))  # strictly increasing: no duplicates, no reordering
    assert [e["payload"]["epoch"] for e in got[:-1]] == list(range(total))  # and nothing lost


async def test_stream_yields_none_keepalive_ticks_while_idle(db, bus, make_run):
    rid = await make_run(status="running")
    stream = stream_run_events(bus, rid, after_seq=0, keepalive_seconds=0.05)
    assert await asyncio.wait_for(stream.__anext__(), 2) is None
    await stream.aclose()
    assert bus.subscriber_count(str(rid)) == 0  # unsubscribed on close


async def test_a_subscriber_that_falls_behind_is_closed_not_fed_lossy_data(bus, monkeypatch):
    monkeypatch.setattr(bus_module, "SUBSCRIBER_QUEUE_SIZE", 3)
    sub = bus.subscribe("r")
    sub.queue = asyncio.Queue(maxsize=3)
    for i in range(5):
        bus.publish("r", {"seq": i})
    assert sub.closed and bus.subscriber_count("r") == 0
    assert (await sub.queue.get()) is bus_module.CLOSED


# -- Log handler -----------------------------------------------------------------------------


class FakeWriter:
    def __init__(self):
        self.lines: list[tuple[str, str, str]] = []

    def log(self, run_id, level, line):
        self.lines.append((run_id, level, line))


@pytest.fixture
def handler():
    fw = FakeWriter()
    # "ludwig" stands in for a trainer backend's log_namespaces (lifespan.py installs these for
    # every installed backend; this fixture predates backends being pluggable).
    h = install_run_log_handler(fw, ("ludwig",))  # type: ignore[arg-type]
    yield h, fw
    uninstall_run_log_handler(h)


def test_handler_only_acts_inside_a_run_context_and_only_on_its_namespaces(handler):
    h, fw = handler
    log = logging.getLogger("ludwig.trainers.trainer")
    log.info("no run context")
    assert fw.lines == []

    token = current_run_id.set("run-1")
    try:
        log.info("epoch 1 done")
        logging.getLogger("theseus.jobs.train").warning("careful")
        logging.getLogger("uvicorn.access").info("GET /health")  # must never be captured
        logging.getLogger("botocore").info("noise")
    finally:
        current_run_id.reset(token)

    assert [(r, lvl, "epoch 1 done" in ln or "careful" in ln) for r, lvl, ln in fw.lines] == [
        ("run-1", "info", True),
        ("run-1", "warn", True),
    ]


def test_handler_follows_the_context_into_a_thread_and_keeps_runs_separate(handler):
    h, fw = handler

    def train(run_id):
        token = current_run_id.set(run_id)
        try:
            ctx = contextvars.copy_context()
            t = threading.Thread(target=ctx.run, args=(logging.getLogger("ludwig.x").info, f"from {run_id}"))
            t.start()
            t.join()
        finally:
            current_run_id.reset(token)

    train("a")
    train("b")
    assert sorted((r, ln.split(": ")[-1]) for r, _, ln in fw.lines) == [("a", "from a"), ("b", "from b")]


def test_handler_tees_every_line_to_the_run_file_even_when_the_live_stream_is_throttled(handler, tmp_path):
    h, fw = handler
    path = tmp_path / "run.log"
    h.attach("r1", path)
    token = current_run_id.set("r1")
    try:
        for i in range(500):  # far above the burst allowance
            logging.getLogger("ludwig.x").info("line %d", i)
    finally:
        current_run_id.reset(token)
        h.detach("r1")

    assert len(path.read_text().splitlines()) == 500  # the file is complete
    streamed = [ln for _, _, ln in fw.lines if "line " in ln]
    assert 0 < len(streamed) < 500  # the live stream was throttled
    assert isinstance(h, RunLogHandler)
