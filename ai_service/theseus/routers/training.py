"""Training runs: list / start / cancel / delete, live events (SSE), evaluation and logs."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from theseus import constants as C
from theseus.backends.registry import describe, list_backends
from theseus.db.models import (
    Dataset,
    DatasetItem,
    DatasetVersion,
    LabelClass,
    RunEvaluation,
    TextFeatures,
    TrainingMetric,
    TrainingRun,
)
from theseus.deps import ProjectDep, RunDep, SessionDep
from theseus.events import get_event_bus, get_event_writer
from theseus.events.stream import stream_run_events
from theseus.jobs import abort
from theseus.schemas.training import (
    ErrorItem,
    EvaluationBrief,
    EvaluationErrorRow,
    EvaluationErrorsResponse,
    EvaluationOut,
    EvaluationResponse,
    MetricRow,
    ModelChoiceOut,
    RunCreatedResponse,
    RunDetail,
    RunDetailResponse,
    RunListResponse,
    RunRow,
    RunStatusResponse,
    RunSummary,
    RunVersion,
    RunVersionDataset,
    TrainBody,
    TrainingBackendListResponse,
    TrainingBackendOut,
)
from theseus.services import storage
from theseus.services.cleanup import cleanup_run_storage
from theseus.services.task_registry import get_task_descriptor
from theseus.services.training import QueueError, queue_training

router = APIRouter(tags=["training"])

ERRORS_PER_PAGE = 50


def _brief(evaluation: RunEvaluation | None) -> EvaluationBrief | None:
    if evaluation is None:
        return None
    return EvaluationBrief(status=evaluation.status, accuracy=evaluation.accuracy, macro_f1=evaluation.macro_f1)


def _backend_out(info) -> TrainingBackendOut:
    return TrainingBackendOut(
        id=info.id,
        label=info.label,
        description=info.description,
        available=info.available,
        unavailable_reason=info.unavailable_reason,
        supported_tasks=info.supported_tasks,
        models=[ModelChoiceOut(**m.model_dump()) for m in info.models],
        model_param_name=info.model_param_name,
        params=info.params,
    )


@router.get("/training-backends", response_model=TrainingBackendListResponse)
async def list_training_backends() -> TrainingBackendListResponse:
    """Every installed trainer backend, whether or not it is currently available (e.g. a missing
    optional dependency) — for a global settings/diagnostics view, not task-scoped model choice."""
    return TrainingBackendListResponse(backends=[_backend_out(describe(b)) for b in list_backends()])


@router.get("/projects/{project_id}/training-backends", response_model=TrainingBackendListResponse)
async def list_project_training_backends(project: ProjectDep) -> TrainingBackendListResponse:
    """Backends that can train this project's task, with their models and hyperparameters — what
    the create-run / create-sweep panel renders. Only available backends are included."""
    task = get_task_descriptor(project.task)
    return TrainingBackendListResponse(
        backends=[
            _backend_out(describe(b, task)) for b in list_backends() if b.available() is None and b.supports(task)
        ]
    )


@router.get("/projects/{project_id}/runs", response_model=RunListResponse)
async def list_runs(project: ProjectDep, session: SessionDep) -> RunListResponse:
    rows = (
        await session.execute(
            sa.select(TrainingRun, RunEvaluation)
            .outerjoin(RunEvaluation, RunEvaluation.run_id == TrainingRun.id)
            .where(TrainingRun.project_id == project.id)
            .order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
        )
    ).all()
    runs = []
    for run, evaluation in rows:
        item = RunSummary.model_validate(run)
        item.evaluation = _brief(evaluation)
        runs.append(item)
    return RunListResponse(runs=runs)


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run: RunDep, session: SessionDep) -> RunDetailResponse:
    version = await session.get(DatasetVersion, run.dataset_version_id)
    dataset = await session.get(Dataset, version.dataset_id)
    metrics = (
        (
            await session.execute(
                sa.select(TrainingMetric)
                .where(TrainingMetric.training_run_id == run.id)
                .order_by(TrainingMetric.epoch, TrainingMetric.split, TrainingMetric.metric_name)
            )
        )
        .scalars()
        .all()
    )
    detail = RunDetail(
        id=run.id,
        name=run.name,
        status=run.status,
        hyperparameters=run.hyperparameters,
        failed_message=run.failed_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
        dataset_version=RunVersion(
            **{c: getattr(version, c) for c in RunVersion.model_fields if c != "dataset"},
            dataset=RunVersionDataset(project_id=version.dataset_id, modality=dataset.modality),
        ),
        metrics=[MetricRow.model_validate(m) for m in metrics],
    )
    return RunDetailResponse(run=detail)


@router.post("/projects/{project_id}/train", response_model=RunCreatedResponse)
async def start_training(body: TrainBody, project: ProjectDep, session: SessionDep) -> RunCreatedResponse:
    version = await session.get(DatasetVersion, body.dataset_version_id)
    if version is None:
        raise HTTPException(404, "Dataset version not found")
    if version.dataset_id != project.id:
        raise HTTPException(400, "Dataset version does not belong to this project")

    result = await queue_training(
        session,
        project_id=project.id,
        name=body.name,
        task=project.task,
        dataset_version_id=body.dataset_version_id,
        backend_id=body.backend,
        hyperparameters=body.hyperparameters,
    )
    if isinstance(result, QueueError):
        raise HTTPException(result.code, result.message)
    return RunCreatedResponse(run=RunRow.model_validate(result))


@router.post("/runs/{run_id}/cancel", status_code=204)
async def cancel_run(run: RunDep) -> None:
    # Only an in-flight run can be canceled. Without this, cancelling a succeeded run flipped it to
    # `canceled`, which permanently blocks inference and export for a model that trained fine.
    if run.status not in ("queued", "running"):
        raise HTTPException(409, f"Cannot cancel a run with status '{run.status}': it has already finished")
    if not await abort.request_cancel(run.id):
        raise HTTPException(409, "Cannot cancel a run that has already finished")


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(run: RunDep, session: SessionDep) -> None:
    """Delete a run whatever its status. An in-flight run is stopped first."""
    if run.status in ("queued", "running"):
        await abort.request_cancel(run.id)
        await get_event_writer().flush()  # so no late event targets the row deleted below
    await cleanup_run_storage(run.id)
    await session.delete(run)
    await session.commit()


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run: RunDep, session: SessionDep) -> RunStatusResponse:
    """Short-poll fallback for the live console."""
    latest = (
        await session.execute(
            sa.select(sa.func.max(TrainingMetric.epoch)).where(TrainingMetric.training_run_id == run.id)
        )
    ).scalar_one()
    metrics = []
    if latest is not None:
        metrics = (
            (
                await session.execute(
                    sa.select(TrainingMetric).where(
                        TrainingMetric.training_run_id == run.id, TrainingMetric.epoch == latest
                    )
                )
            )
            .scalars()
            .all()
        )
    hp = run.hyperparameters if isinstance(run.hyperparameters, dict) else {}
    schedule = hp.get("schedule") if isinstance(hp.get("schedule"), dict) else {}
    epochs_total = hp.get("epochs") or schedule.get("epochs")
    return RunStatusResponse(
        status=run.status,
        epochs_total=epochs_total,
        latest_metrics=[MetricRow.model_validate(m) for m in metrics],
    )


# -- Live events (SSE) -----------------------------------------------------------------------


def _sse(event: dict) -> str:
    return f"id: {event['seq']}\nevent: {event['kind']}\ndata: {json.dumps(event['payload'])}\n\n"


@router.get("/runs/{run_id}/events", include_in_schema=False)
async def stream_events(run: RunDep, session: SessionDep, request: Request) -> StreamingResponse:
    """status / metric / log events, replayable through the standard Last-Event-ID reconnect header.

    The SSE id is the run_events.seq. Documented here rather than in OpenAPI: browsers consume it
    with a native EventSource, which no generated client can model.
    """
    run_id = run.id
    last = request.headers.get("last-event-id", "")
    after_seq = int(last) if last.isdigit() else 0
    # This stream can live for hours. Release the pooled connection NOW: the ownership check left
    # the request session in a transaction, and holding it would exhaust the pool after a handful
    # of open consoles.
    await session.close()

    async def body() -> AsyncIterator[str]:
        yield "retry: 3000\n\n"
        async for event in stream_run_events(get_event_bus(), run_id, after_seq):
            yield ": keepalive\n\n" if event is None else _sse(event)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# -- Evaluation and logs ---------------------------------------------------------------------


@router.get("/runs/{run_id}/evaluation", response_model=EvaluationResponse)
async def get_run_evaluation(run: RunDep, session: SessionDep) -> EvaluationResponse:
    evaluation = await session.get(RunEvaluation, run.id)
    if evaluation is None:
        raise HTTPException(404, "No evaluation report for this run yet")
    return EvaluationResponse(evaluation=EvaluationOut.model_validate(evaluation))


@router.get("/runs/{run_id}/evaluation/errors", response_model=EvaluationErrorsResponse)
async def get_run_evaluation_errors(
    run: RunDep,
    session: SessionDep,
    page: Annotated[int, Query(ge=1)] = 1,
    class_id: Annotated[uuid.UUID | None, Query(alias="classId")] = None,
) -> EvaluationErrorsResponse:
    """Paginated misclassified rows from the report, joined back to their pool items."""
    evaluation = await session.get(RunEvaluation, run.id)
    if evaluation is None or evaluation.status != "success" or not evaluation.report:
        raise HTTPException(404, "No evaluation report for this run yet")

    errors = list((evaluation.report or {}).get("topErrors") or [])
    if class_id is not None:
        # topErrors stores Ludwig own idx2str-derived label STRINGS, never a Postgres class id
        # (label_classes has no idea which index Ludwig assigned to which class), so resolve the
        # class to its name first.
        cls = await session.get(LabelClass, class_id)
        if cls is None:
            raise HTTPException(400, "Unknown label class")
        errors = [e for e in errors if e.get("actual") == cls.name]

    total = len(errors)
    page_errors = errors[(page - 1) * ERRORS_PER_PAGE : page * ERRORS_PER_PAGE]
    item_ids = [uuid.UUID(e["itemId"]) for e in page_errors]
    items: dict[uuid.UUID, tuple[DatasetItem, str | None]] = {}
    if item_ids:
        rows = (
            await session.execute(
                sa.select(DatasetItem, TextFeatures.raw_text)
                .outerjoin(TextFeatures, TextFeatures.item_id == DatasetItem.id)
                .where(DatasetItem.id.in_(item_ids))
            )
        ).all()
        items = {item.id: (item, text) for item, text in rows}

    out = []
    for e in page_errors:
        found = items.get(uuid.UUID(e["itemId"]))
        item_out = None
        if found is not None:
            item, text = found
            url = storage.get_download_url(C.BUCKET_DATASETS, item.storage_url) if item.storage_url else None
            item_out = ErrorItem(id=item.id, text=text, download_url=url)
        out.append(
            EvaluationErrorRow(
                item_id=e["itemId"],
                actual=e["actual"],
                predicted=e["predicted"],
                confidence=e.get("confidence"),
                item=item_out,
            )
        )
    return EvaluationErrorsResponse(errors=out, total=total, page=page, per_page=ERRORS_PER_PAGE)


@router.get("/runs/{run_id}/logs", include_in_schema=False)
async def download_run_logs(run: RunDep) -> RedirectResponse:
    """302 to a presigned URL for the full training log (uploaded whether or not a console was open)."""
    key = storage.training_logs_key(str(run.id))
    exists = await asyncio.get_running_loop().run_in_executor(None, storage.file_exists, C.BUCKET_TRAINING, key)
    if not exists:
        raise HTTPException(404, "No log file for this run")
    return RedirectResponse(storage.get_download_url(C.BUCKET_TRAINING, key), status_code=302)
