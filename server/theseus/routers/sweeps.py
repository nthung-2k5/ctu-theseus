"""Hyperparameter sweeps, orchestrated on top of the ordinary training pipeline.

A sweep is a search space expanded into N trials, each an ordinary training_runs row (sweep_id +
trial_index) enqueued through queue_training. It has no execution engine of its own (see
services/sweep.py for why this is not Ludwig hyperopt).
"""

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from theseus.db.models import DatasetVersion, RunEvaluation, Sweep, TrainingRun
from theseus.deps import ProjectDep, SessionDep, SweepDep
from theseus.schemas.training import (
    CreateSweepBody,
    EvaluationBrief,
    RunRow,
    SweepCreatedResponse,
    SweepDetailResponse,
    SweepListResponse,
    SweepRow,
    SweepSummary,
    SweepTrial,
)
from theseus.services import training as training_service
from theseus.services.training import QueueError, queue_sweep, reconcile_sweep_status

router = APIRouter(tags=["sweeps"])


@router.post("/projects/{project_id}/sweeps", status_code=201, response_model=SweepCreatedResponse)
async def create_sweep(body: CreateSweepBody, project: ProjectDep, session: SessionDep) -> SweepCreatedResponse:
    version = await session.get(DatasetVersion, body.dataset_version_id)
    if version is None:
        raise HTTPException(404, "Dataset version not found")
    if version.dataset_id != project.id:
        raise HTTPException(400, "Dataset version does not belong to this project")

    result = await queue_sweep(
        session,
        project_id=project.id,
        name=body.name,
        task=project.task,
        dataset_version_id=body.dataset_version_id,
        backend_id=body.backend,
        search_space=body.search_space,
        strategy=body.strategy,
        max_trials=body.max_trials,
    )
    if isinstance(result, QueueError):
        raise HTTPException(result.code, result.message)
    sweep, trials = result
    return SweepCreatedResponse(sweep=SweepRow.model_validate(sweep), trials=[RunRow.model_validate(t) for t in trials])


@router.get("/projects/{project_id}/sweeps", response_model=SweepListResponse)
async def list_sweeps(project: ProjectDep, session: SessionDep) -> SweepListResponse:
    sweeps = (
        (
            await session.execute(
                sa.select(Sweep)
                .where(Sweep.project_id == project.id)
                .order_by(Sweep.created_at.desc(), Sweep.id.desc())
            )
        )
        .scalars()
        .all()
    )
    trial_rows = (
        await session.execute(
            sa.select(TrainingRun.sweep_id, TrainingRun.status).where(TrainingRun.sweep_id.in_([s.id for s in sweeps]))
        )
    ).all()
    total: dict = {}
    done: dict = {}
    for sweep_id, status in trial_rows:
        total[sweep_id] = total.get(sweep_id, 0) + 1
        if status not in ("queued", "running"):
            done[sweep_id] = done.get(sweep_id, 0) + 1
    return SweepListResponse(
        sweeps=[
            SweepSummary(
                id=s.id,
                name=s.name,
                strategy=s.strategy,
                max_trials=s.max_trials,
                status=s.status,
                created_at=s.created_at,
                trial_count=total.get(s.id, 0),
                completed_trial_count=done.get(s.id, 0),
            )
            for s in sweeps
        ]
    )


@router.get("/sweeps/{sweep_id}", response_model=SweepDetailResponse)
async def get_sweep(sweep: SweepDep, session: SessionDep) -> SweepDetailResponse:
    sweep = await reconcile_sweep_status(session, sweep)
    rows = (
        await session.execute(
            sa.select(TrainingRun, RunEvaluation)
            .outerjoin(RunEvaluation, RunEvaluation.run_id == TrainingRun.id)
            .where(TrainingRun.sweep_id == sweep.id)
            .order_by(TrainingRun.trial_index)
        )
    ).all()
    trials = []
    for run, evaluation in rows:
        trial = SweepTrial.model_validate(run)
        if evaluation is not None:
            trial.evaluation = EvaluationBrief(
                status=evaluation.status, accuracy=evaluation.accuracy, macro_f1=evaluation.macro_f1
            )
        trials.append(trial)
    return SweepDetailResponse(sweep=SweepRow.model_validate(sweep), trials=trials)


@router.post("/sweeps/{sweep_id}/cancel", status_code=204)
async def cancel_sweep(sweep: SweepDep, session: SessionDep) -> None:
    if sweep.status != "running":
        raise HTTPException(409, f"Cannot cancel a sweep with status '{sweep.status}': it has already finished")
    await training_service.cancel_sweep(session, sweep.id)
