"""/api/v1/predict: the hosted prediction API, authenticated by bearer API key (not a cookie).

This is a public contract for third-party scripts, so it is tagged separately from the
session-authenticated surface. Every route is rate limited per key (see deps.current_api_key_user).
Dispatch and polling are shared with routers/inference.py through services/inference.py, so the
two surfaces cannot drift on validation.
"""

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from theseus import constants as C
from theseus.deps import ApiKeyRunDep, SessionDep
from theseus.jobs import inference as inference_jobs
from theseus.jobs.dispatcher import get_dispatcher
from theseus.routers.inference import task_of
from theseus.schemas.serving import InferenceAccepted, PolledJob, SyncPredictResponse
from theseus.services import inference as svc
from theseus.services import storage
from theseus.settings import get_settings

router = APIRouter(prefix="/v1/predict", tags=["v1"])

FileField = Annotated[UploadFile | None, File()]
FieldsField = Annotated[str | None, Form()]
TopKField = Annotated[int | None, Form(alias="topK", ge=1, le=1000)]


@router.post("/{run_id}", status_code=202, response_model=InferenceAccepted)
async def v1_predict(
    run: ApiKeyRunDep, session: SessionDep, file: FileField = None, fields: FieldsField = None, top_k: TopKField = None
) -> InferenceAccepted:
    result = await svc.dispatch_inference(
        session, run, await task_of(session, run), file=file, fields=fields, top_k=top_k
    )
    if isinstance(result, svc.DispatchError):
        raise HTTPException(result.code, result.message)
    return InferenceAccepted(inference_id=result.inference_id)


@router.post(
    "/{run_id}/sync",
    response_model=SyncPredictResponse,
    response_model_exclude_none=True,
    responses={
        202: {"model": SyncPredictResponse, "description": "Not finished in time: poll the returned inferenceId"}
    },
)
async def v1_predict_sync(
    run: ApiKeyRunDep, session: SessionDep, file: FileField = None, fields: FieldsField = None, top_k: TopKField = None
):
    """Dispatch and wait for the result.

    A real await rather than a poll loop. asyncio.shield keeps the job running when the wait
    times out or the client disconnects: on timeout the caller gets the same 202 + inferenceId as
    before and can poll, and a disconnect can never kill a GPU job another poller may be watching.
    """
    dispatcher = get_dispatcher()
    if dispatcher is not None and not dispatcher.has_capacity("inference"):
        # A synchronous request would otherwise queue behind a cold model load with the connection held open.
        raise HTTPException(503, "The inference workers are busy, retry shortly", headers={"Retry-After": "5"})

    result = await svc.dispatch_inference(
        session, run, await task_of(session, run), file=file, fields=fields, top_k=top_k, sync=True
    )
    if isinstance(result, svc.DispatchError):
        raise HTTPException(result.code, result.message)

    try:
        await asyncio.wait_for(asyncio.shield(result.waiter), get_settings().sync_predict_max_wait_seconds)
    except TimeoutError:
        inference_jobs.drop_waiter(result.inference_id)
        pending = SyncPredictResponse(inference_id=result.inference_id, status="pending")
        return JSONResponse(pending.model_dump(mode="json", by_alias=True, exclude_none=True), status_code=202)

    job = await svc.get_job(session, run.id, result.inference_id, refresh=True)
    return SyncPredictResponse(inference_id=result.inference_id, **svc.polled(job))


@router.post("/{run_id}/batch", status_code=202, response_model=InferenceAccepted)
async def v1_predict_batch(
    run: ApiKeyRunDep, session: SessionDep, file: Annotated[UploadFile, File()]
) -> InferenceAccepted:
    result = await svc.dispatch_batch_inference(session, run, await task_of(session, run), file)
    if isinstance(result, svc.DispatchError):
        raise HTTPException(result.code, result.message)
    return InferenceAccepted(inference_id=result.inference_id)


@router.get("/{run_id}/jobs/{inference_id}", response_model=PolledJob, response_model_exclude_none=True)
async def v1_get_prediction_job(run: ApiKeyRunDep, inference_id: uuid.UUID, session: SessionDep) -> PolledJob:
    job = await svc.get_job(session, run.id, inference_id)
    if job is None:
        raise HTTPException(404, "Inference job not found for this run")
    return PolledJob(**svc.polled(job))


@router.get("/{run_id}/jobs/{inference_id}/download", include_in_schema=False)
async def v1_download_prediction_result(
    run: ApiKeyRunDep, inference_id: uuid.UUID, session: SessionDep
) -> RedirectResponse:
    key = svc.batch_result_key(await svc.get_job(session, run.id, inference_id))
    if key is None:
        raise HTTPException(404, "Inference job not found for this run, or has no downloadable result")
    return RedirectResponse(storage.get_download_url(C.BUCKET_MODELS, key), status_code=302)
