"""Model exports in three tiers (model / devkit / app), assembled as a zip bundle.

POST /runs/{run_id}/exports inserts a `pending` row and returns 202. One export job (jobs/export.py)
then walks pending -> converting -> assembling -> ready | failed, converting the model artifact only
if an earlier export of another tier or language has not already produced it. There is no
lazy-reconcile-on-GET: a crashed job is re-queued by lease expiry or startup recovery.
"""

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from theseus import constants as C
from theseus.db.enums import APP_TARGETS, DEVKIT_LANGS
from theseus.db.models import ModelExport, Project
from theseus.deps import ExportDep, RunDep, SessionDep
from theseus.jobs.dispatcher import nudge
from theseus.schemas.serving import CreateExportBody, ExportAccepted, ExportListResponse, ExportResponse, ExportRow
from theseus.services import storage

router = APIRouter(tags=["export"])


@router.post("/runs/{run_id}/exports", status_code=202, response_model=ExportAccepted)
async def create_export(body: CreateExportBody, run: RunDep, session: SessionDep) -> ExportAccepted:
    if run.status != "succeeded":
        raise HTTPException(409, f"Run is not succeeded (status: {run.status})")

    if body.tier != "model":
        # devkit / app ship an ONNX client: TorchScript would need libtorch at runtime, which would
        # make a portable client a lie.
        if body.format != "onnx":
            raise HTTPException(400, f"tier '{body.tier}' only supports format 'onnx'")
        if not body.lang:
            raise HTTPException(400, f"tier '{body.tier}' requires a lang")
        valid = DEVKIT_LANGS if body.tier == "devkit" else APP_TARGETS
        if body.lang not in valid:
            raise HTTPException(400, f"tier '{body.tier}' requires lang to be one of: {', '.join(valid)}")

    project = await session.get(Project, run.project_id)
    row = ModelExport(
        run_id=run.id,
        user_id=project.user_id,
        tier=body.tier,
        format=body.format,
        lang=None if body.tier == "model" else body.lang,
        status="pending",
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    nudge("export")
    return ExportAccepted(export_id=row.id)


@router.get("/runs/{run_id}/exports", response_model=ExportListResponse)
async def list_exports(run: RunDep, session: SessionDep) -> ExportListResponse:
    rows = (
        (
            await session.execute(
                sa.select(ModelExport)
                .where(ModelExport.run_id == run.id)
                .order_by(ModelExport.created_at.desc(), ModelExport.id.desc())
            )
        )
        .scalars()
        .all()
    )
    return ExportListResponse(exports=[ExportRow.model_validate(r) for r in rows])


@router.get("/exports/{export_id}", response_model=ExportResponse)
async def get_export(export: ExportDep) -> ExportResponse:
    return ExportResponse(export=ExportRow.model_validate(export))


@router.get("/exports/{export_id}/download", include_in_schema=False)
async def download_export(export: ExportDep) -> RedirectResponse:
    """302 to a presigned URL for the assembled bundle."""
    if export.status != "ready" or not export.bundle_key:
        raise HTTPException(404, "Export is not ready for download")
    return RedirectResponse(storage.get_download_url(C.BUCKET_MODELS, export.bundle_key), status_code=302)
