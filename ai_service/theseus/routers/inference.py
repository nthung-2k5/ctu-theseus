"""/api/inference: session-authenticated inference.

A prediction runs inside the request and its result is the response: nothing is queued or stored (see
services/inference.py). Body shape depends on the run task (see get_inference_input_spec): file-backed
tasks send `file`; text and tabular tasks send `fields`, a JSON-encoded object (multipart form fields
cannot carry nested objects). Batch scoring takes a CSV and answers with the scored CSV.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from theseus.db.models import Project
from theseus.deps import RunDep, SessionDep
from theseus.schemas.serving import PredictResponse
from theseus.services import inference as svc
from theseus.services.inference import spawn_warm

router = APIRouter(prefix="/inference", tags=["inference"])

FileField = Annotated[UploadFile | None, File()]
FieldsField = Annotated[str | None, Form()]
TopKField = Annotated[int | None, Form(alias="topK", ge=1, le=1000)]

BATCH_CSV_RESPONSES = {
    200: {
        "description": "The uploaded rows plus the prediction columns. `X-Row-Count` holds the row count.",
        "content": {"text/csv": {"schema": {"type": "string"}}},
    }
}


async def task_of(session, run) -> str:
    project = await session.get(Project, run.project_id)
    if project is None:
        raise HTTPException(404, "Run not found")
    return project.task


def csv_response(result: svc.BatchResult) -> Response:
    return Response(
        result.csv,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predictions.csv"',
            "X-Row-Count": str(result.row_count),
            "Access-Control-Expose-Headers": "X-Row-Count",
        },
    )


@router.post("/{run_id}", response_model=PredictResponse)
async def run_inference(
    run: RunDep, session: SessionDep, file: FileField = None, fields: FieldsField = None, top_k: TopKField = None
) -> PredictResponse:
    output = await svc.predict_one(run, await task_of(session, run), file=file, fields=fields, top_k=top_k)
    return PredictResponse(output=output)


@router.post("/{run_id}/batch", response_class=Response, responses=BATCH_CSV_RESPONSES)
async def run_batch_inference(run: RunDep, session: SessionDep, file: Annotated[UploadFile, File()]) -> Response:
    """One CSV row per prediction, scored in a single predict call."""
    return csv_response(await svc.predict_batch(run, await task_of(session, run), file))


@router.post("/{run_id}/warm", status_code=202)
async def warm_inference_model(run: RunDep) -> None:
    """Preload the run model into the cache before the user first request (fire and forget)."""
    if run.status != "succeeded":
        raise HTTPException(409, "No successfully trained model found for this run")
    spawn_warm(str(run.id))
    await asyncio.sleep(0)  # let the warm task start before the response returns
