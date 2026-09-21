"""Startup recovery: settle every job a previous process left in flight.

Runs after the event writer starts and BEFORE the dispatchers do, so nothing new is claimed
until the old state is reconciled. With exactly one process, a restart is unambiguous: whatever
is marked in-flight in the database is definitely not running any more.

  training   running -> failed. NEVER auto-resumed: that is what max_deliver=1 meant. A restarted
             run must not silently re-burn hours of GPU. Queued runs are untouched (they never
             started, so they are simply picked up).
  export     converting/assembling -> pending (assembly rewrites the same S3 keys, so re-running
             is idempotent), or failed once attempts are exhausted.
  inference  running -> pending if attempts remain and the input still exists, else failed. A
             synchronous request whose temp file died with the process cannot be re-run.
  snapshot   building -> failed (the parquet build is in-process and cannot resume).

This also retires the old heartbeat/ack_wait drift bug: the reaper and JetStream used to disagree
about when a silent run was dead, letting a run flip failed -> running -> succeeded.
"""

import logging
import os
from dataclasses import dataclass

import sqlalchemy as sa

from theseus.db.base import get_sessionmaker
from theseus.db.models import DatasetVersion, InferenceJob, ModelExport, TrainingRun
from theseus.events import get_event_writer
from theseus.jobs import inference as inference_jobs

logger = logging.getLogger(__name__)

RESTART_MESSAGE = "Server restarted during the job"


@dataclass
class RecoveryReport:
    runs_failed: int = 0
    exports_requeued: int = 0
    exports_failed: int = 0
    inference_requeued: int = 0
    inference_failed: int = 0
    snapshots_failed: int = 0


def _input_available(job: InferenceJob) -> bool:
    payload = job.payload or {}
    if payload.get("kind") not in ("file", "batch"):
        return True  # text / record payloads live entirely on the row
    if job.upload_key:
        return True
    local = payload.get("localPath")
    return bool(local and os.path.exists(local))


async def recover_on_startup() -> RecoveryReport:
    report = RecoveryReport()
    sessionmaker = get_sessionmaker()

    # Training: fail through the writer, so the CAS guard applies and an SSE client is told.
    async with sessionmaker() as s:
        run_ids = (
            (
                await s.execute(
                    sa.update(TrainingRun)
                    .where(TrainingRun.status == "running")
                    .values(claimed_by=None, lease_expires_at=None, last_error=RESTART_MESSAGE)
                    .returning(TrainingRun.id)
                )
            )
            .scalars()
            .all()
        )
        await s.commit()
    writer = get_event_writer()
    for run_id in run_ids:
        writer.status(run_id, "failed", "Server restarted during training")
    await writer.flush()
    report.runs_failed = len(run_ids)

    async with sessionmaker() as s:
        in_flight = ("converting", "assembling")
        requeued = (
            (
                await s.execute(
                    sa.update(ModelExport)
                    .where(ModelExport.status.in_(in_flight), ModelExport.attempt < ModelExport.max_attempts)
                    .values(
                        status="pending",
                        claimed_by=None,
                        lease_expires_at=None,
                        available_at=sa.func.now(),
                        last_error=RESTART_MESSAGE,
                    )  # fmt: skip
                    .returning(ModelExport.id)
                )
            )
            .scalars()
            .all()
        )
        failed = (
            (
                await s.execute(
                    sa.update(ModelExport)
                    .where(ModelExport.status.in_(in_flight))
                    .values(
                        status="failed",
                        claimed_by=None,
                        lease_expires_at=None,
                        failed_message=RESTART_MESSAGE,
                        last_error=RESTART_MESSAGE,
                    )  # fmt: skip
                    .returning(ModelExport.id)
                )
            )
            .scalars()
            .all()
        )
        report.exports_requeued, report.exports_failed = len(requeued), len(failed)

        jobs = (await s.execute(sa.select(InferenceJob).where(InferenceJob.status == "running"))).scalars().all()
        retry = [j.id for j in jobs if j.attempt < j.max_attempts and _input_available(j)]
        retry_set = set(retry)
        dead = [j.id for j in jobs if j.id not in retry_set]
        if retry:
            await s.execute(
                sa.update(InferenceJob)
                .where(InferenceJob.id.in_(retry), InferenceJob.status == "running")
                .values(status="pending", claimed_by=None, lease_expires_at=None, available_at=sa.func.now())
            )
        if dead:
            await s.execute(
                sa.update(InferenceJob)
                .where(InferenceJob.id.in_(dead), InferenceJob.status == "running")
                .values(
                    status="failed",
                    claimed_by=None,
                    lease_expires_at=None,
                    last_error=RESTART_MESSAGE,
                    error=inference_jobs.GENERIC_FAILURE,
                    completed_at=sa.func.now(),
                )  # fmt: skip
            )
        report.inference_requeued, report.inference_failed = len(retry), len(dead)

        snapshots = (
            (
                await s.execute(
                    sa.update(DatasetVersion)
                    .where(DatasetVersion.status == "building")
                    .values(status="failed", failed_message="Server restarted while building the snapshot")
                    .returning(DatasetVersion.id)
                )
            )
            .scalars()
            .all()
        )
        report.snapshots_failed = len(snapshots)
        await s.commit()

    for job_id in dead:
        await inference_jobs.cleanup_upload(job_id)

    logger.info("Startup recovery: %s", report)
    return report
