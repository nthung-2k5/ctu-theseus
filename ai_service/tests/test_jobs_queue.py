import asyncio
from datetime import timedelta

import sqlalchemy as sa

from theseus.db.models import ModelExport, TrainingRun
from theseus.jobs import queue


async def get(db, model, id_):
    async with db() as s:
        return (await s.execute(sa.select(model).where(model.id == id_))).scalar_one()


async def test_claim_moves_the_oldest_queued_row_to_running_and_records_attempt_and_lease(db, make_run):
    first = await make_run()
    second = await make_run()

    claimed = await queue.claim_one(queue.TRAIN, "worker-1", 300)
    assert claimed == first  # oldest first (uuidv7 ids and created_at both order by time)
    run = await get(db, TrainingRun, first)
    assert (run.status, run.attempt, run.claimed_by) == ("running", 1, "worker-1")
    assert run.lease_expires_at is not None
    assert (await get(db, TrainingRun, second)).status == "queued"


async def test_claim_returns_none_when_nothing_is_queued_or_yet_available(db, make_run):
    assert await queue.claim_one(queue.TRAIN, "w", 300) is None
    rid = await make_run()
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun).where(TrainingRun.id == rid).values(available_at=sa.func.now() + timedelta(hours=1))
        )
        await s.commit()
    assert await queue.claim_one(queue.TRAIN, "w", 300) is None  # backed off: not available yet


async def test_concurrent_claims_never_hand_the_same_job_to_two_workers(db, make_run):
    ids = {await make_run() for _ in range(3)}
    results = await asyncio.gather(*(queue.claim_one(queue.TRAIN, f"w{i}", 300) for i in range(10)))

    claimed = [r for r in results if r is not None]
    assert len(claimed) == 3 and set(claimed) == ids  # every job claimed, none twice
    assert len(set(claimed)) == len(claimed)


async def test_claim_only_considers_its_own_kind_and_status(db, make_run, make_export):
    await make_run(status="running")  # not queued
    export_id, _ = await make_export()
    assert await queue.claim_one(queue.TRAIN, "w", 300) is None
    assert await queue.claim_one(queue.EXPORT, "w", 300) == export_id
    assert (await get(db, ModelExport, export_id)).status == "converting"  # export claim state


async def test_renew_lease_extends_only_an_in_flight_job(db, make_run):
    rid = await make_run()
    await queue.claim_one(queue.TRAIN, "w", 10)
    before = (await get(db, TrainingRun, rid)).lease_expires_at
    assert await queue.renew_lease(queue.TRAIN, rid, 600) is True
    assert (await get(db, TrainingRun, rid)).lease_expires_at > before

    finished = await make_run(status="succeeded")
    assert await queue.renew_lease(queue.TRAIN, finished, 600) is False


async def test_release_or_fail_requeues_with_a_delay_until_attempts_are_exhausted(db, make_export):
    job_id, _ = await make_export()
    # attempt 1 of 3 fails: back to pending, delayed
    await queue.claim_one(queue.EXPORT, "w", 300)
    assert await queue.release_or_fail(queue.EXPORT, job_id, "boom 1") == "pending"
    job = await get(db, ModelExport, job_id)
    assert (job.status, job.claimed_by, job.last_error) == ("pending", None, "boom 1")
    assert await queue.claim_one(queue.EXPORT, "w", 300) is None  # 30s backoff: not claimable yet

    async with db() as s:
        await s.execute(sa.update(ModelExport).values(available_at=sa.func.now()))
        await s.commit()
    await queue.claim_one(queue.EXPORT, "w", 300)  # attempt 2
    assert await queue.release_or_fail(queue.EXPORT, job_id, "boom 2") == "pending"
    async with db() as s:
        await s.execute(sa.update(ModelExport).values(available_at=sa.func.now()))
        await s.commit()
    await queue.claim_one(queue.EXPORT, "w", 300)  # attempt 3 = max
    final = {"failed_message": "generic"}
    assert await queue.release_or_fail(queue.EXPORT, job_id, "boom 3", final) == "failed"
    job = await get(db, ModelExport, job_id)
    assert (job.status, job.failed_message, job.attempt) == ("failed", "generic", 3)


async def test_final_values_are_not_applied_when_the_job_is_only_being_retried(db, make_export):
    job_id, _ = await make_export()
    await queue.claim_one(queue.EXPORT, "w", 300)
    await queue.release_or_fail(queue.EXPORT, job_id, "boom", {"failed_message": "generic"})
    job = await get(db, ModelExport, job_id)
    assert job.status == "pending" and job.failed_message is None


async def test_train_jobs_fail_immediately_because_max_attempts_is_one(db, make_run):
    rid = await make_run()
    await queue.claim_one(queue.TRAIN, "w", 300)
    assert await queue.release_or_fail(queue.TRAIN, rid, "oom") == "failed"


async def test_release_or_fail_is_a_noop_for_a_job_no_longer_in_flight(db, make_export):
    job_id, _ = await make_export(status="ready")
    assert await queue.release_or_fail(queue.EXPORT, job_id, "late failure") is None
    assert (await get(db, ModelExport, job_id)).status == "ready"
