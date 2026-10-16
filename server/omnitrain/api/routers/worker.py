"""/api/worker: what a remote training worker (`python -m omnitrain.worker`) calls, authenticated by its bearer token.

The worker pulls: it asks for a run, trains it on its own machine against S3, and reports back here. This module is
the only place a remote run touches the database, and it does so through the same two things local training uses, the
event writer (the single writer of run events) and the evaluation store, so every invariant of local runs holds:
late events after a terminal status are dropped, and a status change is a guarded compare-and-swap.

  POST /worker/claim                   next queued run for this worker, 204 when there is none
  POST /worker/runs/{id}/events        a batch of metrics and a heartbeat; the answer says whether to stop
  POST /worker/runs/{id}/complete      succeeded / failed / canceled, with the evaluation on success
"""

from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Response

from omnitrain import __version__
from omnitrain.api.deps import SessionDep, WorkerDep, WorkerRunDep
from omnitrain.core import constants as C
from omnitrain.core.executors import run_in_executor
from omnitrain.db.enums import TERMINAL_RUN_STATUSES
from omnitrain.db.models import CustomModel, TrainingRun, TrainingWorker
from omnitrain.events import get_event_writer
from omnitrain.jobs import queue
from omnitrain.schemas.worker import (
    PROTOCOL_VERSION,
    WorkerClaimBody,
    WorkerCompleteBody,
    WorkerCompleteResponse,
    WorkerCustomModel,
    WorkerEvaluation,
    WorkerEventsBody,
    WorkerEventsResponse,
    WorkerRunSpec,
)
from omnitrain.services import custom_models, storage
from omnitrain.services.run_evaluation_store import record_evaluation
from omnitrain.settings import get_settings

router = APIRouter(prefix="/worker", tags=["worker"])


async def _custom_model(session: SessionDep, run: TrainingRun) -> WorkerCustomModel | None:
    """The run's bring-your-own model, described for a worker. The claim already only picks a run whose model a worker
    can fetch on its own (queue._worker_can_take_model), so this is a description, not a decision."""
    if run.custom_model_id is None:
        return None
    row = await session.get(CustomModel, run.custom_model_id)
    assert row is not None  # training_runs.custom_model_id is ON DELETE RESTRICT
    return WorkerCustomModel(
        id=row.id,
        kind=row.kind,
        source_kind=row.source_kind,  # type: ignore[arg-type]
        source_ref=row.source_ref,
        revision=row.revision,
        sha256=row.sha256,
        spec=dict(row.spec or {}),
        server_path=custom_models.ref_for(row).local_path,
    )


def _evaluation_values(run_id: str, evaluation: WorkerEvaluation) -> dict[str, Any]:
    """The `run_evaluations` columns for what a worker reported. Object keys come from the run id, never the worker."""
    if evaluation.status == "failed":
        return {"status": "failed", "failed_message": evaluation.failed_message}
    return {
        "status": "success",
        "split": evaluation.split,
        "report_key": storage.evaluation_report_key(run_id),
        "predictions_key": storage.evaluation_predictions_key(run_id),
        "report": evaluation.report,
        "accuracy": evaluation.accuracy,
        "macro_f1": evaluation.macro_f1,
        "failed_message": None,
    }


@router.post(
    "/claim", response_model=WorkerRunSpec, responses={204: {"description": "No run is waiting for this worker"}}
)
async def worker_claim(body: WorkerClaimBody, worker: WorkerDep, session: SessionDep) -> WorkerRunSpec | Response:
    """Hand this worker the oldest queued run it can train, or 204.

    The worker's self-description is stored first, even when nothing is handed out: the queue reads it to decide
    whether the server should wait for this worker. A worker running a different package version advertises no
    backends (so the server does not wait for it) and is told why.
    """
    mismatch = body.protocol != PROTOCOL_VERSION
    info = {
        "backends": [] if mismatch else sorted(set(body.backends)),
        "hostname": body.hostname,
        "gpu": body.gpu,
        "version": body.version,
        "protocol": body.protocol,
    }
    await session.execute(sa.update(TrainingWorker).where(TrainingWorker.id == worker.id).values(info=info))
    await session.commit()
    if mismatch:
        raise HTTPException(
            409,
            f"This worker speaks worker protocol {body.protocol} (omnitrain {body.version}) but the server speaks "
            f"{PROTOCOL_VERSION} (omnitrain {__version__}): run the same commit of omnitrain on both",
        )
    if not worker.enabled:
        return Response(status_code=204)

    run_id = await queue.claim_for_worker(worker.id, info["backends"], get_settings().job_lease_seconds)
    if run_id is None:
        return Response(status_code=204)

    writer = get_event_writer()
    run = await session.get(TrainingRun, run_id)
    assert run is not None
    if run.cancel_requested_at is not None:
        # Canceled between being queued and being claimed: the same early exit run_train takes locally.
        writer.status(run_id, "canceled")
        return Response(status_code=204)
    writer.status(run_id, "running")
    return WorkerRunSpec(
        id=run.id,
        backend=run.backend,
        config=run.config or {},
        dataset_version_id=run.dataset_version_id,
        config_key=run.config_key,
        custom_model=await _custom_model(session, run),
    )


@router.post("/runs/{run_id}/events", response_model=WorkerEventsResponse)
async def worker_events(body: WorkerEventsBody, run: WorkerRunDep) -> WorkerEventsResponse:
    """Metrics and a heartbeat from a training worker. The answer is how the worker learns of a cancel."""
    terminal = run.status in TERMINAL_RUN_STATUSES
    if not terminal:
        writer = get_event_writer()
        for m in body.metrics:
            writer.metric(run.id, m.epoch, m.split, m.metrics)
        if body.heartbeat:
            writer.touch(run.id)
    return WorkerEventsResponse(cancel=run.cancel_requested_at is not None, terminal=terminal)


@router.post("/runs/{run_id}/complete", response_model=WorkerCompleteResponse)
async def worker_complete(body: WorkerCompleteBody, run: WorkerRunDep) -> WorkerCompleteResponse:
    """The worker's last word on a run. Idempotent: a run that already ended keeps the status it has."""
    if run.status in TERMINAL_RUN_STATUSES:
        return WorkerCompleteResponse(status=run.status)

    writer = get_event_writer()
    run_id = str(run.id)
    if body.status == "succeeded":
        results = await run_in_executor(
            None, storage.list_keys, C.BUCKET_TRAINING, storage.training_results_prefix(run_id)
        )
        if not results:
            # Success with nothing to serve would leave a "succeeded" run nobody can export or predict with.
            message = "The worker reported success, but no results were found in storage"
            writer.status(run.id, "failed", message)
            raise HTTPException(409, message)
        if body.evaluation is not None:
            await record_evaluation(run.id, _evaluation_values(run_id, body.evaluation))
        writer.status(run.id, "succeeded")
    elif body.status == "failed":
        writer.status(run.id, "failed", body.message or "The worker reported a failure")
    else:
        writer.status(run.id, "canceled")
    return WorkerCompleteResponse(status=body.status)
