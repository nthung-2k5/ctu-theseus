"""Model exports, assembled as a zip bundle by a pluggable export format.

GET /export-formats lists the installed formats (theseus/export/formats/, one class each).
POST /runs/{run_id}/exports inserts a `pending` row and returns 202. One export job (jobs/export.py)
then walks pending -> converting -> assembling -> ready | failed, converting the model artifact only
if an earlier export of another format has not already produced it. There is no
lazy-reconcile-on-GET: a crashed job is re-queued by lease expiry or startup recovery.
"""

import uuid
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from theseus import constants as C
from theseus.backends.registry import get_backend
from theseus.db.models import ModelExport, Project
from theseus.deps import ExportDep, RunDep, SessionDep, UserId, owned_run
from theseus.export.registry import find_export_format
from theseus.export.registry import list_export_formats as installed_export_formats
from theseus.jobs.dispatcher import nudge
from theseus.schemas.serving import (
    CreateExportBody,
    ExportAccepted,
    ExportFormatListResponse,
    ExportFormatOut,
    ExportListResponse,
    ExportResponse,
    ExportRow,
)
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor

router = APIRouter(tags=["export"])


@router.get("/export-formats", response_model=ExportFormatListResponse)
async def list_export_formats(
    user_id: UserId, session: SessionDep, run_id: Annotated[uuid.UUID | None, Query(alias="runId")] = None
) -> ExportFormatListResponse:
    """Installed export formats, grouped and ordered for display.

    With `runId`, only the formats that support that run's project task AND whose artifact the
    run's own trainer backend can actually produce.
    """
    task = None
    backend = None
    if run_id is not None:
        run = await owned_run(run_id, user_id, session)
        project = await session.get(Project, run.project_id)
        task = get_task_descriptor(project.task)
        backend = get_backend(run.backend)
    return ExportFormatListResponse(
        formats=[
            ExportFormatOut(
                id=f.id,
                label=f.label,
                description=f.description,
                notice=f.notice,
                group=f.group,
                artifact=f.artifact,
            )
            for f in installed_export_formats(task, backend)
        ]
    )


@router.post("/runs/{run_id}/exports", status_code=202, response_model=ExportAccepted)
async def create_export(body: CreateExportBody, run: RunDep, session: SessionDep) -> ExportAccepted:
    if run.status != "succeeded":
        raise HTTPException(409, f"Run is not succeeded (status: {run.status})")

    export_format = find_export_format(body.format)
    if export_format is None:
        raise HTTPException(400, f"Unknown export format '{body.format}'")

    project = await session.get(Project, run.project_id)
    if not export_format.supports(get_task_descriptor(project.task)):
        raise HTTPException(400, f"Export format '{body.format}' does not support this project's task")
    backend = get_backend(run.backend)
    if export_format.artifact not in backend.artifacts:
        raise HTTPException(400, f"Export format '{body.format}' is not available for a run trained by '{backend.id}'")

    row = ModelExport(run_id=run.id, user_id=project.user_id, format=body.format, status="pending")
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
