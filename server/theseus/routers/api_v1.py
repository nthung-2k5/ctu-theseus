"""/api/v1/predict: the hosted prediction API, authenticated by bearer API key (not a cookie).

This is a public contract for third-party scripts, so it is tagged separately from the
session-authenticated surface. Every route is rate limited per key (see deps.current_api_key_user).
Predictions run inside the request and are answered directly; nothing is queued or stored. The
session routes in routers/inference.py share the same service, so the two surfaces cannot drift on
validation.
"""

from typing import Annotated

from fastapi import APIRouter, File, Form, Response, UploadFile

from theseus.deps import ApiKeyRunDep, SessionDep
from theseus.routers.inference import BATCH_CSV_RESPONSES, csv_response, task_of
from theseus.schemas.serving import PredictResponse
from theseus.services import inference as svc

router = APIRouter(prefix="/v1/predict", tags=["v1"])

FileField = Annotated[UploadFile | None, File()]
FieldsField = Annotated[str | None, Form()]
TopKField = Annotated[int | None, Form(alias="topK", ge=1, le=1000)]


# `/sync` is the old name of this route (when plain POST queued a job and answered 202). Kept as a hidden
# alias so existing scripts keep working.
@router.post("/{run_id}/sync", response_model=PredictResponse, include_in_schema=False)
@router.post("/{run_id}", response_model=PredictResponse)
async def v1_predict(
    run: ApiKeyRunDep, session: SessionDep, file: FileField = None, fields: FieldsField = None, top_k: TopKField = None
) -> PredictResponse:
    """Score one item and return the prediction. 503 + Retry-After when every inference slot is busy."""
    output = await svc.predict_one(run, await task_of(session, run), file=file, fields=fields, top_k=top_k)
    return PredictResponse(output=output)


@router.post("/{run_id}/batch", response_class=Response, responses=BATCH_CSV_RESPONSES)
async def v1_predict_batch(run: ApiKeyRunDep, session: SessionDep, file: Annotated[UploadFile, File()]) -> Response:
    """Score every row of a CSV (text and tabular tasks) and return the scored CSV."""
    return csv_response(await svc.predict_batch(run, await task_of(session, run), file))
