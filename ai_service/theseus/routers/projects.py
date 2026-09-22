"""/api/projects: CRUD. Creating a project also creates its 1:1 dataset and mutable draft version."""

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from theseus.db.models import Dataset, DatasetVersion, DatasetVersionItem, LabelClass, Project, TrainingRun
from theseus.deps import ProjectDep, SessionDep, UserId
from theseus.events import get_event_writer
from theseus.jobs import abort
from theseus.schemas.projects import (
    CreateProjectBody,
    DatasetOut,
    DraftDatasetOut,
    LabelClassOut,
    ProjectDetail,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectResponse,
    ProjectRow,
    ProjectSummary,
    UpdateProjectBody,
)
from theseus.services.cleanup import cleanup_project_storage
from theseus.services.dataset_views import split_counts_for_dataset, version_with_counts
from theseus.services.task_registry import get_task_descriptor, task_to_modality

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ProjectListResponse)
async def list_projects(user_id: UserId, session: SessionDep) -> ProjectListResponse:
    rows = (
        await session.execute(
            sa.select(Project, Dataset.modality)
            .outerjoin(Dataset, Dataset.project_id == Project.id)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.desc(), Project.id.desc())
        )
    ).all()
    return ProjectListResponse(
        projects=[
            ProjectSummary(
                id=p.id,
                name=p.name,
                description=p.description,
                task=p.task,
                created_at=p.created_at,
                draft_dataset=DraftDatasetOut(modality=modality) if modality else None,
            )
            for p, modality in rows
        ]
    )


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project: ProjectDep, session: SessionDep) -> ProjectDetailResponse:
    version_count = (
        await session.execute(sa.select(sa.func.count()).where(DatasetVersion.dataset_id == project.id))
    ).scalar_one()
    run_count = (
        await session.execute(sa.select(sa.func.count()).where(TrainingRun.project_id == project.id))
    ).scalar_one()
    dataset = await session.get(Dataset, project.id)

    dataset_out = None
    if dataset is not None:
        versions = (
            (
                await session.execute(
                    sa.select(DatasetVersion)
                    .where(DatasetVersion.dataset_id == project.id)
                    .order_by(DatasetVersion.created_at, DatasetVersion.id)
                )
            )
            .scalars()
            .all()
        )
        classes = (
            (
                await session.execute(
                    sa.select(LabelClass)
                    .where(LabelClass.dataset_id == project.id, LabelClass.is_active.is_(True))
                    .order_by(LabelClass.created_at, LabelClass.class_id)
                )
            )
            .scalars()
            .all()
        )
        counts = await split_counts_for_dataset(session, project.id)
        views = [version_with_counts(v, counts) for v in versions]
        dataset_out = DatasetOut(
            project_id=dataset.project_id,
            modality=dataset.modality,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            draft=next((v for v in views if v.version_tag is None), None),
            # Snapshots only: the mutable draft has its own field, and every consumer of `versions`
            # (the Snapshots page, the sidebar count, the training pickers) means immutable snapshots.
            versions=[v for v in views if v.version_tag is not None],
            classes=[LabelClassOut.model_validate(c) for c in classes],
        )

    detail = ProjectDetail(
        **ProjectRow.model_validate(project).model_dump(),
        run_count=run_count,
        version_count=version_count,
        dataset=dataset_out,
    )
    return ProjectDetailResponse(project=detail)


@router.post("", response_model=ProjectResponse)
async def create_project(body: CreateProjectBody, user_id: UserId, session: SessionDep) -> ProjectResponse:
    descriptor = get_task_descriptor(body.task)
    if descriptor.backend != "ludwig":
        raise HTTPException(422, f'Task "{body.task}" is not yet trainable ({descriptor.status}).')

    # All three rows commit together: a project with no dataset (or a dataset with no draft
    # version) is unusable, so a failure partway must not leave any of them behind.
    project = Project(user_id=user_id, name=body.name, description=body.description, task=body.task)
    session.add(project)
    await session.flush()
    session.add(Dataset(project_id=project.id, modality=task_to_modality(body.task)))
    await session.flush()
    session.add(DatasetVersion(dataset_id=project.id))  # version_tag NULL = the draft
    await session.commit()
    await session.refresh(project)
    return ProjectResponse(project=ProjectRow.model_validate(project))


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(body: UpdateProjectBody, project: ProjectDep, session: SessionDep) -> ProjectResponse:
    for field in body.model_fields_set:
        value = getattr(body, field)
        if field == "name" and value is None:
            continue  # name is required in the database, so an explicit null means leave it
        setattr(project, field, value)
    await session.commit()
    await session.refresh(project)
    return ProjectResponse(project=ProjectRow.model_validate(project))


@router.delete("/{project_id}", status_code=204)
async def delete_project(project: ProjectDep, session: SessionDep) -> None:
    """Cascades to dataset, versions, items and runs; S3 objects are removed best-effort first."""
    # A run still training would otherwise keep writing events for a row that no longer exists.
    active = (
        (
            await session.execute(
                sa.select(TrainingRun.id).where(
                    TrainingRun.project_id == project.id, TrainingRun.status.in_(("queued", "running"))
                )
            )
        )
        .scalars()
        .all()
    )
    for run_id in active:
        await abort.request_cancel(run_id)
    if active:
        await get_event_writer().flush()

    # Delete S3 objects before the cascade removes the rows whose keys they are.
    await cleanup_project_storage(session, project.id)
    # Snapshot membership rows hold RESTRICT foreign keys to items, so remove them first.
    versions = sa.select(DatasetVersion.id).where(DatasetVersion.dataset_id == project.id)
    await session.execute(sa.delete(DatasetVersionItem).where(DatasetVersionItem.version_id.in_(versions)))
    await session.delete(project)
    await session.commit()
