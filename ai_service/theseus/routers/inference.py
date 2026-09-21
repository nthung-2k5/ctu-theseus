"""/api/inference: session-authenticated inference (dispatch is asynchronous, results are polled).

Dispatch inserts a `pending` inference_jobs row and returns 202 + inferenceId; the job runs on the
inference lane and writes its result straight to that row, so the poll route just reads Postgres.
Body shape depends on the run task (see get_inference_input_spec): file-backed tasks send `file`;
text and tabular tasks send `fields`, a JSON-encoded object (multipart form fields cannot carry
nested objects).
"""

import asyncio
import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from theseus import constants as C
from theseus.db.models import InferenceJob, Project
from theseus.deps import RunDep, SessionDep
from theseus.jobs.inference import spawn_warm
from theseus.schemas.serving import InferenceAccepted, InferenceJobListResponse, InferenceJobRow, PolledJob
from theseus.services import inference as svc
from theseus.services import storage

router = APIRouter(prefix="/inference", tags=["inference"])

FileField = Annotated[UploadFile | None, File()]
FieldsField = Annotated[str | None, Form()]
TopKField = Annotated[int | None, Form(alias="topK", ge=1, le=1000)]


async def task_of(session, run) -> str:
    project = await session.get(Project, run.project_id)
    if project is None:
        raise HTTPException(404, "Run not found")
    return project.task


@router.post("/{run_id}", status_code=202, response_model=InferenceAccepted)
async def run_inference(
    run: RunDep, session: SessionDep, file: FileField = None, fields: FieldsField = None, top_k: TopKField = None
) -> InferenceAccepted:
    result = await svc.dispatch_inference(
        session, run, await task_of(session, run), file=file, fields=fields, top_k=top_k
    )
    if isinstance(result, svc.DispatchError):
        raise HTTPException(result.code, result.message)
    return InferenceAccepted(inference_id=result.inference_id)


@router.post("/{run_id}/batch", status_code=202, response_model=InferenceAccepted)
async def run_batch_inference(
    run: RunDep, session: SessionDep, file: Annotated[UploadFile, File()]
) -> InferenceAccepted:
    """One CSV row per prediction, scored in a single Ludwig predict call."""
    result = await svc.dispatch_batch_inference(session, run, await task_of(session, run), file)
    if isinstance(result, svc.DispatchError):
        raise HTTPException(result.code, result.message)
    return InferenceAccepted(inference_id=result.inference_id)


@router.get("/{run_id}/jobs", response_model=InferenceJobListResponse)
async def list_inference_jobs(run: RunDep, session: SessionDep) -> InferenceJobListResponse:
    jobs = (
        (
            await session.execute(
                sa.select(InferenceJob)
                .where(InferenceJob.run_id == run.id)
                .order_by(InferenceJob.created_at.desc(), InferenceJob.id.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    return InferenceJobListResponse(
        jobs=[
            InferenceJobRow(
                id=j.id,
                run_id=j.run_id,
                status=svc.public_status(j.status),
                output=j.output,
                error=j.error,
                created_at=j.created_at,
                completed_at=j.completed_at,
            )
            for j in jobs
        ]
    )


@router.get("/{run_id}/jobs/{inference_id}", response_model=PolledJob, response_model_exclude_none=True)
async def get_inference_job(run: RunDep, inference_id: uuid.UUID, session: SessionDep) -> PolledJob:
    job = await svc.get_job(session, run.id, inference_id)
    if job is None:
        raise HTTPException(404, "Inference job not found for this run")
    return PolledJob(**svc.polled(job))


@router.get("/{run_id}/jobs/{inference_id}/download", include_in_schema=False)
async def download_inference_result(run: RunDep, inference_id: uuid.UUID, session: SessionDep) -> RedirectResponse:
    key = svc.batch_result_key(await svc.get_job(session, run.id, inference_id))
    if key is None:
        raise HTTPException(404, "Inference job not found for this run, or has no downloadable result")
    return RedirectResponse(storage.get_download_url(C.BUCKET_MODELS, key), status_code=302)


@router.post("/{run_id}/warm", status_code=202)
async def warm_inference_model(run: RunDep) -> None:
    """Preload the run model into the cache before the user first request (fire and forget)."""
    if run.status != "succeeded":
        raise HTTPException(409, "No successfully trained model found for this run")
    spawn_warm(str(run.id))
    await asyncio.sleep(0)  # let the warm task start before the response returns
