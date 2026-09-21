"""Enqueue training runs and sweeps (ported from queueTraining / queueSweep in lib/microservice.ts).

Enqueueing is just: compile the Ludwig config, put it in S3, insert a `queued` row, nudge the
train lane. There is no message to publish: the dispatcher claims the row (see jobs/queue.py).
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
import uuid_utils
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.db.models import DatasetVersion, Sweep, TrainingRun
from theseus.events import get_event_writer
from theseus.jobs import abort
from theseus.jobs.dispatcher import nudge
from theseus.services import storage
from theseus.services.ludwig_config import (
    ConfigError,
    TrainerSelections,
    compile_ludwig_config,
    serialize_ludwig_config,
)
from theseus.services.snapshot import read_snapshot_manifest
from theseus.services.sweep import expand_sweep, validate_search_space
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)


@dataclass
class QueueError:
    code: int
    message: str


def new_uuid7() -> uuid.UUID:
    """Time-ordered id generated app-side, because the S3 config key needs the run id before the insert."""
    return uuid.UUID(str(uuid_utils.uuid7()))


async def queue_training(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    name: str,
    task: str,
    dataset_version_id: uuid.UUID,
    selections: TrainerSelections | None = None,
    sweep_id: uuid.UUID | None = None,
    trial_index: int | None = None,
    record_compile_failure: bool = False,
) -> TrainingRun | QueueError:
    """Compile, upload the config and insert a queued run.

    record_compile_failure is for sweep trials: a trial whose config cannot compile (say an
    encoder id that slipped past validation) is stored as a failed run so it shows up in the
    leaderboard, rather than silently vanishing and leaving a sweep with fewer trials than asked.
    """
    version = await session.get(DatasetVersion, dataset_version_id)
    if version is None:
        return QueueError(404, "Dataset version not found")
    if version.status != "ready":
        return QueueError(409, f"Dataset version is not ready for training (status: {version.status})")

    sel = selections or TrainerSelections()
    run_id = new_uuid7()
    hyperparameters = sel.model_dump(by_alias=True, exclude_none=True)
    loop = asyncio.get_running_loop()

    try:
        ctx = await read_snapshot_manifest(dataset_version_id)
        ludwig_config = compile_ludwig_config(get_task_descriptor(task), ctx, sel)
        config_yaml = serialize_ludwig_config(ludwig_config)
    except (ConfigError, ValueError, KeyError) as e:
        message = f"Failed to compile Ludwig config: {e}"
        if not record_compile_failure:
            return QueueError(400, message)
        run = TrainingRun(
            id=run_id, project_id=project_id, name=name, dataset_version_id=dataset_version_id, sweep_id=sweep_id,
            trial_index=trial_index, hyperparameters=hyperparameters, status="failed", failed_message=message,
            completed_at=sa.func.now(),
        )  # fmt: skip
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run

    config_key = storage.training_config_key(str(run_id))
    await loop.run_in_executor(
        None, storage.upload_bytes, C.BUCKET_TRAINING, config_key, config_yaml.encode(), "application/yaml"
    )

    run = TrainingRun(
        id=run_id, project_id=project_id, name=name, dataset_version_id=dataset_version_id, sweep_id=sweep_id,
        trial_index=trial_index, hyperparameters=hyperparameters, ludwig_config=ludwig_config, config_key=config_key,
        status="queued",
    )  # fmt: skip
    session.add(run)
    await session.commit()
    await session.refresh(run)

    get_event_writer().status(run_id, "queued")
    nudge("train")
    return run


async def queue_sweep(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    name: str,
    task: str,
    dataset_version_id: uuid.UUID,
    search_space: dict[str, list[Any]],
    strategy: str,
    max_trials: int,
) -> tuple[Sweep, list[TrainingRun]] | QueueError:
    """Expand a search space into trials and enqueue each through the ordinary training path.

    A sweep has no execution engine of its own: the train lane runs one job at a time in FIFO
    order, which is exactly what serializes the trials on the single GPU.
    """
    version = await session.get(DatasetVersion, dataset_version_id)
    if version is None:
        return QueueError(404, "Dataset version not found")
    if version.status != "ready":
        return QueueError(409, f"Dataset version is not ready for training (status: {version.status})")
    if error := validate_search_space(search_space, max_trials):
        return QueueError(400, error)

    trials_selections = expand_sweep(search_space, strategy, max_trials)  # type: ignore[arg-type]
    sweep = Sweep(
        project_id=project_id, dataset_version_id=dataset_version_id, name=name, search_space=search_space,
        strategy=strategy, max_trials=max_trials, status="running",
    )  # fmt: skip
    session.add(sweep)
    await session.commit()
    await session.refresh(sweep)

    trials: list[TrainingRun] = []
    for index, raw in enumerate(trials_selections):
        result = await queue_training(
            session,
            project_id=project_id,
            name=f"{name} — trial {index + 1}",
            task=task,
            dataset_version_id=dataset_version_id,
            selections=TrainerSelections.model_validate(raw),
            sweep_id=sweep.id,
            trial_index=index,
            record_compile_failure=True,
        )
        if isinstance(result, TrainingRun):
            trials.append(result)
    return sweep, trials


async def cancel_sweep(session: AsyncSession, sweep_id: uuid.UUID) -> None:
    """Cancel a sweep and every trial that has not finished."""
    active = (
        (
            await session.execute(
                sa.select(TrainingRun.id).where(
                    TrainingRun.sweep_id == sweep_id, TrainingRun.status.in_(("queued", "running"))
                )
            )
        )
        .scalars()
        .all()
    )
    for run_id in active:
        await abort.request_cancel(run_id)
    await session.execute(
        sa.update(Sweep).where(Sweep.id == sweep_id, Sweep.status == "running").values(status="canceled")
    )
    await session.commit()


async def reconcile_sweep_status(session: AsyncSession, sweep: Sweep) -> Sweep:
    """Bring a running sweep to `completed` once every trial reached a terminal status by itself.

    Computed on read rather than by a reaper: nothing time-sensitive depends on a sweep flipping
    the instant its last trial finishes.
    """
    if sweep.status != "running":
        return sweep
    unfinished = (
        await session.execute(
            sa.select(TrainingRun.id)
            .where(TrainingRun.sweep_id == sweep.id, TrainingRun.status.in_(("queued", "running")))
            .limit(1)
        )
    ).first()
    if unfinished is not None:
        return sweep
    await session.execute(
        sa.update(Sweep).where(Sweep.id == sweep.id, Sweep.status == "running").values(status="completed")
    )
    await session.commit()
    await session.refresh(sweep)
    return sweep
