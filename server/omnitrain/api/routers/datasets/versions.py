"""Snapshot versions: create, read and delete, and the preprocessing/augmentation options a snapshot offers."""

import asyncio

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from omnitrain.api.deps import DraftDep, ProjectDep, SessionDep, VersionDep
from omnitrain.db.models import (
    Dataset,
    DatasetVersion,
    DatasetVersionItem,
)
from omnitrain.domain.task_registry import get_task_descriptor
from omnitrain.plugins.augmentation.config import MAX_AUGMENTED_ITEMS
from omnitrain.plugins.augmentation.registry import (
    AugmentationConfigError,
    describe,
    list_augmentations,
    validate_config,
)
from omnitrain.plugins.preprocessing.config import MAX_PREPROCESSED_ITEMS
from omnitrain.plugins.preprocessing.registry import (
    PreprocessingConfigError,
    list_preprocessing,
)
from omnitrain.plugins.preprocessing.registry import (
    describe as describe_preprocessing,
)
from omnitrain.plugins.preprocessing.registry import (
    validate_config as validate_preprocessing_config,
)
from omnitrain.schemas.datasets import (
    AugmentationListResponse,
    CreateVersionBody,
    PreprocessingListResponse,
    VersionCreatedResponse,
    VersionDataset,
    VersionDetail,
    VersionDetailResponse,
)
from omnitrain.schemas.projects import DatasetSplit, VersionOut
from omnitrain.services import datasets as svc
from omnitrain.services.cleanup import cleanup_version_storage
from omnitrain.services.dataset_views import split_counts_for_dataset
from omnitrain.services.derived_items import delete_derived_items
from omnitrain.services.snapshot import build_snapshot

router = APIRouter()

_background: set[asyncio.Task] = set()


@router.post("/projects/{project_id}/versions", status_code=202, response_model=VersionCreatedResponse)
async def create_version(body: CreateVersionBody, draft: DraftDep, session: SessionDep) -> VersionCreatedResponse:
    """Snapshot the draft current pool membership into a new immutable version (built asynchronously).

    With `preprocessing`, the build first replaces items with a deterministic preprocessed copy for
    whichever splits each op selects (see services/preprocessing.py). With `augmentation`, the build
    then also materializes augmented copies of the TRAIN split, made from any preprocessed
    replacement (see services/augmentation.py); validation and test stay real so metrics measure
    real data.
    """
    tag = body.version_tag

    draft_count = (
        await session.execute(sa.select(sa.func.count()).where(DatasetVersionItem.version_id == draft.draft.id))
    ).scalar_one()
    if draft_count == 0:
        raise HTTPException(400, "The draft is empty; add items before creating a snapshot")

    preprocessing_config = None
    if body.preprocessing is not None:
        try:
            pre_config = validate_preprocessing_config(get_task_descriptor(draft.project.task), body.preprocessing)
        except PreprocessingConfigError as e:
            raise HTTPException(400, str(e)) from None
        selected_splits = {s for op in pre_config.ops for s in op.splits}
        selected_count = (
            await session.execute(
                sa.select(sa.func.count()).where(
                    DatasetVersionItem.version_id == draft.draft.id,
                    DatasetVersionItem.split_type.in_(selected_splits),
                )
            )
        ).scalar_one()
        if selected_count == 0:
            raise HTTPException(400, "The draft has no items in the selected splits to preprocess")
        if selected_count > MAX_PREPROCESSED_ITEMS:
            raise HTTPException(
                400,
                f"{selected_count} items in the selected splits exceeds the limit of "
                f"{MAX_PREPROCESSED_ITEMS} preprocessed items per snapshot",
            )
        preprocessing_config = pre_config.model_dump(by_alias=True, exclude_none=True)

    augmentation_config = None
    if body.augmentation is not None:
        try:
            config = validate_config(get_task_descriptor(draft.project.task), body.augmentation)
        except AugmentationConfigError as e:
            raise HTTPException(400, str(e)) from None
        train_count = (
            await session.execute(
                sa.select(sa.func.count()).where(
                    DatasetVersionItem.version_id == draft.draft.id, DatasetVersionItem.split_type == "train"
                )
            )
        ).scalar_one()
        if train_count == 0:
            raise HTTPException(400, "The draft has no training items to augment")
        if train_count * config.copies_per_item > MAX_AUGMENTED_ITEMS:
            raise HTTPException(
                400,
                f"{train_count} training items x {config.copies_per_item} copies exceeds the limit of "
                f"{MAX_AUGMENTED_ITEMS} augmented items per snapshot",
            )
        augmentation_config = config.model_dump(by_alias=True, exclude_none=True)

    version = DatasetVersion(
        dataset_id=draft.project.id,
        version_tag=tag,
        status="building",
        preprocessing_config=preprocessing_config,
        augmentation_config=augmentation_config,
    )
    session.add(version)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, f"A version tagged '{tag}' already exists") from None

    await session.execute(
        pg_insert(DatasetVersionItem).from_select(
            ["version_id", "item_id", "split_type"],
            sa.select(sa.literal(version.id), DatasetVersionItem.item_id, DatasetVersionItem.split_type).where(
                DatasetVersionItem.version_id == draft.draft.id
            ),
        )
    )
    version.item_count = (
        await session.execute(sa.select(sa.func.count()).where(DatasetVersionItem.version_id == version.id))
    ).scalar_one()
    await session.commit()
    await session.refresh(version)

    # Fire and forget: the parquet is metadata-sized (S3 URIs, not bytes) so building inline is cheap.
    # Held in a strong set (the loop keeps only weak references); startup recovery fails any version
    # left `building` by a restart.
    task = asyncio.create_task(build_snapshot(version.id))
    _background.add(task)
    task.add_done_callback(_background.discard)
    return VersionCreatedResponse(version=VersionOut.model_validate(version))


@router.get("/versions/{version_id}", response_model=VersionDetailResponse)
async def get_version(version: VersionDep, session: SessionDep) -> VersionDetailResponse:
    counts = (await split_counts_for_dataset(session, version.dataset_id)).get(version.id)
    dataset = await session.get(Dataset, version.dataset_id)
    detail = VersionDetail(
        **VersionOut.model_validate(version).model_dump(exclude={"splits"}),
        splits=[DatasetSplit(split_type=s, item_count=getattr(counts, s, 0) if counts else 0) for s in svc.SPLITS],
        dataset=VersionDataset(project_id=dataset.project_id, modality=dataset.modality),
    )
    return VersionDetailResponse(version=detail)


@router.delete("/versions/{version_id}", status_code=204)
async def delete_version(version: VersionDep, session: SessionDep) -> None:
    if version.version_tag is None:
        raise HTTPException(400, "Cannot delete draft version")
    await cleanup_version_storage(version.id, version.version_tag)
    # Preprocessed replacements and augmented copies belong to this snapshot alone: their rows go
    # with it (membership first, see delete_derived_items), unlike pool items which other snapshots
    # and the draft may share.
    await delete_derived_items(session, version.id)
    await session.delete(version)
    await session.commit()


@router.get("/projects/{project_id}/preprocessing", response_model=PreprocessingListResponse)
async def list_preprocessing_options(project: ProjectDep) -> PreprocessingListResponse:
    """The preprocessing ops a snapshot of this project can be built with (installed ops that support its task)."""
    task = get_task_descriptor(project.task)
    return PreprocessingListResponse(preprocessing=[describe_preprocessing(op) for op in list_preprocessing(task)])


@router.get("/projects/{project_id}/augmentations", response_model=AugmentationListResponse)
async def list_augmentation_options(project: ProjectDep) -> AugmentationListResponse:
    """The augmentations a snapshot of this project can be built with (installed op classes that support its task)."""
    task = get_task_descriptor(project.task)
    return AugmentationListResponse(augmentations=[describe(op) for op in list_augmentations(task)])
