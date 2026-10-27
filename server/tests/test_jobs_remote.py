"""Training on a remote worker, queue side: who may claim a run, when the server holds back for a worker, and what a
restart or a dead worker does to a run that is not the server's.

The worker-facing HTTP routes and the worker process come in later phases; these tests cover only the claim rules
(`queue.claim_for_worker`, the `claim_one(TRAIN)` guard), startup recovery and the heartbeat reaper.
"""

import asyncio
import uuid
from datetime import timedelta

import pytest
import sqlalchemy as sa

from omnitrain.db.models import CustomModel, TrainingRun, TrainingWorker
from omnitrain.events import InProcessRunEventBus, set_event_writer
from omnitrain.events.writer import EventWriter
from omnitrain.jobs import queue, reapers
from omnitrain.jobs.recovery import recover_on_startup


@pytest.fixture
async def writer(db):
    w = EventWriter(InProcessRunEventBus())
    await w.start()
    set_event_writer(w)
    yield w
    await w.stop()
    set_event_writer(None)


async def run_row(db, run_id):
    async with db() as s:
        return (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()


async def update_worker(db, worker_id, **values):
    async with db() as s:
        await s.execute(sa.update(TrainingWorker).where(TrainingWorker.id == worker_id).values(**values))
        await s.commit()


async def age_run(db, run_id, seconds=3600):
    """Make a run look like it has been queued for a while (past the remote-claim grace period)."""
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == run_id)
            .values(created_at=sa.func.now() - timedelta(seconds=seconds))
        )
        await s.commit()


async def make_custom_model(db, **fields) -> uuid.UUID:
    values = {"backend": "ludwig", "kind": "hf_transformer", "name": "m", "source_kind": "hub", "source_ref": "a/b"}
    async with db() as s:
        row = CustomModel(**{**values, **fields})
        s.add(row)
        await s.commit()
        return row.id


# -- A worker's claim ------------------------------------------------------------------------


async def test_a_worker_claims_a_run_and_the_run_records_where_it_trains(db, make_run, make_worker):
    worker = await make_worker()
    rid = await make_run()

    assert await queue.claim_for_worker(worker, ["ludwig"], 300) == rid

    run = await run_row(db, rid)
    assert (run.status, run.attempt, run.worker_id) == ("running", 1, worker)
    assert run.claimed_by == f"worker:{worker}"


async def test_a_worker_only_takes_runs_for_backends_it_offers_and_never_a_custom_model_run(db, make_run, make_worker):
    worker = await make_worker(("ludwig",))
    other_backend = await make_run(backend="yolo")
    custom = await make_run(custom_model_id=await make_custom_model(db))
    assert await queue.claim_for_worker(worker, ["ludwig"], 300) is None

    plain = await make_run()  # younger than the other two, but the only one it may take
    assert await queue.claim_for_worker(worker, ["ludwig"], 300) == plain
    assert (await run_row(db, other_backend)).status == "queued"
    assert (await run_row(db, custom)).status == "queued"


async def test_a_worker_trains_one_run_at_a_time(db, make_run, make_worker):
    worker = await make_worker()
    first = await make_run()
    second = await make_run()

    assert await queue.claim_for_worker(worker, ["ludwig"], 300) == first
    assert await queue.claim_for_worker(worker, ["ludwig"], 300) is None
    assert (await run_row(db, second)).status == "queued"

    async with db() as s:  # it finished: now the next one
        await s.execute(sa.update(TrainingRun).where(TrainingRun.id == first).values(status="succeeded"))
        await s.commit()
    assert await queue.claim_for_worker(worker, ["ludwig"], 300) == second


async def test_one_worker_polling_twice_at_once_still_gets_a_single_run(db, make_run, make_worker):
    worker = await make_worker()
    for _ in range(3):
        await make_run()

    results = await asyncio.gather(*(queue.claim_for_worker(worker, ["ludwig"], 300) for _ in range(6)))

    assert len([r for r in results if r is not None]) == 1


@pytest.mark.parametrize("change", [{"enabled": False}, {"revoked_at": sa.func.now()}])
async def test_a_disabled_or_revoked_worker_claims_nothing(db, make_run, make_worker, change):
    worker = await make_worker()
    rid = await make_run()
    await update_worker(db, worker, **change)

    assert await queue.claim_for_worker(worker, ["ludwig"], 300) is None
    assert (await run_row(db, rid)).status == "queued"


async def test_an_unknown_worker_claims_nothing(db, make_run):
    await make_run()
    assert await queue.claim_for_worker(uuid.uuid4(), ["ludwig"], 300) is None


# -- Bring-your-own models ---------------------------------------------------------------------

# (label, fields of the custom model, whether a worker may train a video run on it)
VIDEO_MODELS = [
    ("upload", {"source_kind": "upload", "sha256": "a" * 64}, True),
    ("public_hub", {"source_kind": "hub", "revision": "c0ffee", "spec": {"private": False}}, True),
    ("hub_never_checked_for_privacy", {"source_kind": "hub", "revision": "c0ffee"}, True),
    ("private_hub", {"source_kind": "hub", "revision": "c0ffee", "spec": {"private": True}}, False),
]
MODEL_CASES = pytest.mark.parametrize(
    ("fields", "allowed"), [(f, a) for _, f, a in VIDEO_MODELS], ids=[n for n, _, _ in VIDEO_MODELS]
)


@MODEL_CASES
async def test_a_worker_takes_a_custom_model_run_only_if_it_can_fetch_the_model_itself(
    db, make_run, make_worker, fields, allowed
):
    worker = await make_worker(("video",))
    rid = await make_run(backend="video", custom_model_id=await make_custom_model(db, backend="video", **fields))

    claimed = await queue.claim_for_worker(worker, ["video"], 300, custom_backends=["video"])

    assert claimed == (rid if allowed else None)
    assert (await run_row(db, rid)).status == ("running" if allowed else "queued")


async def test_a_backend_that_did_not_opt_in_keeps_its_custom_model_runs_on_the_server(db, make_run, make_worker):
    worker = await make_worker(("ludwig",))
    rid = await make_run(custom_model_id=await make_custom_model(db, source_kind="upload", sha256="a" * 64))

    assert await queue.claim_for_worker(worker, ["ludwig"], 300, custom_backends=["video"]) is None
    assert (await run_row(db, rid)).status == "queued"


def test_the_installed_backends_decide_who_opted_in_and_only_video_pytorch_and_llm_did():
    # Ludwig's checkpoint re-reads its model from the path in its config, so it stays on the server.
    assert sorted(queue.remote_custom_backends()) == ["llm", "pytorch", "video"]


@MODEL_CASES
async def test_the_server_waits_for_a_worker_on_a_custom_model_run_exactly_when_the_worker_could_take_it(
    db, make_run, make_worker, fields, allowed
):
    await make_worker(("video",))
    rid = await make_run(backend="video", custom_model_id=await make_custom_model(db, backend="video", **fields))

    claimed = await queue.claim_one(queue.TRAIN, "host:1", 300)

    assert claimed == (None if allowed else rid)


# -- The server's claim, and when it waits for a worker ---------------------------------------


async def test_the_server_holds_a_fresh_run_back_for_an_idle_online_worker_then_takes_it(db, make_run, make_worker):
    await make_worker()
    rid = await make_run()

    assert await queue.claim_one(queue.TRAIN, "host:1", 300) is None  # the worker gets first refusal
    assert (await run_row(db, rid)).status == "queued"

    await age_run(db, rid)  # the worker never took it: the server does
    assert await queue.claim_one(queue.TRAIN, "host:1", 300) == rid
    run = await run_row(db, rid)
    assert (run.status, run.worker_id, run.claimed_by) == ("running", None, "host:1")


@pytest.mark.parametrize(
    "why",
    ["offline", "disabled", "revoked", "other_backend", "busy", "custom_model"],
)
async def test_the_server_does_not_wait_when_no_worker_could_take_the_run(db, make_run, make_worker, why):
    worker_backends = ("yolo",) if why == "other_backend" else ("ludwig",)
    worker = await make_worker(worker_backends)
    fields = {}
    if why == "offline":
        await update_worker(db, worker, last_seen_at=sa.func.now() - timedelta(minutes=10))
    elif why == "disabled":
        await update_worker(db, worker, enabled=False)
    elif why == "revoked":
        await update_worker(db, worker, revoked_at=sa.func.now())
    elif why == "busy":
        await make_run(status="running", worker_id=worker)
    elif why == "custom_model":
        fields["custom_model_id"] = await make_custom_model(db)
    rid = await make_run(**fields)

    assert await queue.claim_one(queue.TRAIN, "host:1", 300) == rid


async def test_a_worker_that_never_reported_in_does_not_hold_the_server_back(db, make_run, make_worker):
    await make_worker(last_seen_at=None)
    rid = await make_run()
    assert await queue.claim_one(queue.TRAIN, "host:1", 300) == rid


async def test_the_server_and_a_worker_racing_for_one_run_produce_one_winner(db, make_run, make_worker):
    worker = await make_worker()
    rid = await make_run()
    await age_run(db, rid)  # both are now allowed to take it

    server, remote = await asyncio.gather(
        queue.claim_one(queue.TRAIN, "host:1", 300), queue.claim_for_worker(worker, ["ludwig"], 300)
    )

    assert [server, remote].count(rid) == 1 and [server, remote].count(None) == 1
    run = await run_row(db, rid)
    assert run.attempt == 1
    assert (run.worker_id is not None) == (remote == rid)


# -- Restart and a dead worker --------------------------------------------------------------------


async def test_a_server_restart_ends_its_own_runs_but_leaves_a_remote_run_going(db, writer, make_run, make_worker):
    worker = await make_worker()
    local = await make_run(status="running")
    remote = await make_run(status="running", worker_id=worker)

    report = await recover_on_startup()

    assert report.runs_failed == 1
    assert (await run_row(db, local)).status == "failed"
    assert (await run_row(db, remote)).status == "running"


async def test_the_heartbeat_reaper_fails_a_remote_run_whose_worker_went_silent(db, writer, make_run, make_worker):
    worker = await make_worker()
    silent = await make_run(status="running", worker_id=worker)
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == silent)
            .values(heartbeat_at=sa.func.now() - timedelta(hours=1))
        )
        await s.commit()

    assert await reapers.stale_runs(timeout_seconds=900) == 1
    await writer.flush()
    assert (await run_row(db, silent)).status == "failed"


async def test_deleting_a_worker_keeps_the_runs_it_trained(db, make_run, make_worker):
    worker = await make_worker()
    rid = await make_run(status="succeeded", worker_id=worker)

    async with db() as s:
        await s.execute(sa.delete(TrainingWorker).where(TrainingWorker.id == worker))
        await s.commit()

    run = await run_row(db, rid)
    assert (run.status, run.worker_id) == ("succeeded", None)
