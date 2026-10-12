"""The dataset pool: snapshot versions, items, uploads, splits, labeling, health, annotations.

A project has one dataset (1:1). Items live in a project-wide, content-addressed, deduplicated pool
(dataset_items); a version (the mutable draft, or an immutable snapshot) is just a set of pool items
with a split assignment (dataset_version_items). Snapshotting copies pool-item membership into a new
version and kicks off an async parquet build (services/snapshot.py); clients poll GET /versions/{id}
for status ready | failed.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal

import sqlalchemy as sa
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain import constants as C
from omnitrain.augmentation.config import MAX_AUGMENTED_ITEMS
from omnitrain.augmentation.registry import AugmentationConfigError, describe, list_augmentations, validate_config
from omnitrain.db.chunks import chunked
from omnitrain.db.models import (
    Annotation,
    AudioFeatures,
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    TabularFeatures,
    TextFeatures,
    VideoFeatures,
    VisionFeatures,
)
from omnitrain.deps import AnnotationDep, DraftDep, ItemDep, ProjectDep, SessionDep, VersionDep
from omnitrain.preprocessing.config import MAX_PREPROCESSED_ITEMS
from omnitrain.preprocessing.registry import (
    PreprocessingConfigError,
    list_preprocessing,
)
from omnitrain.preprocessing.registry import (
    describe as describe_preprocessing,
)
from omnitrain.preprocessing.registry import (
    validate_config as validate_preprocessing_config,
)
from omnitrain.schemas.datasets import (
    AnnotationListResponse,
    AnnotationOut,
    AnnotationResponse,
    AudioFeaturesOut,
    AudioHealth,
    AugmentationListResponse,
    AutoSplitBody,
    AutoSplitCounts,
    AutoSplitResponse,
    ClassCount,
    ClassifyBody,
    ClassifyResponse,
    CreateAnnotationBody,
    CreateItemsBody,
    CreateItemsResponse,
    CreateVersionBody,
    DatasetHealth,
    DatasetHealthResponse,
    DeleteItemsBody,
    DeleteItemsResponse,
    DeleteOutcome,
    HealthClassCount,
    ItemFailure,
    ItemListResponse,
    ItemOut,
    ItemRow,
    ItemRowWithDuplicate,
    ItemSelection,
    MinMaxAvg,
    PreprocessingListResponse,
    SetSplitBody,
    SetSplitResponse,
    TabularFeaturesOut,
    TabularHealth,
    TextFeaturesOut,
    TextHealth,
    UpdateAnnotationBody,
    UploadResponse,
    UploadResult,
    VersionCreatedResponse,
    VersionDataset,
    VersionDetail,
    VersionDetailResponse,
    VideoFeaturesOut,
    VideoHealth,
    VisionFeaturesOut,
    VisionHealth,
)
from omnitrain.schemas.projects import DatasetSplit, VersionOut
from omnitrain.services import datasets as svc
from omnitrain.services import storage, thumbnails
from omnitrain.services import video as video_media
from omnitrain.services.boxes import is_background, validate_box_annotation
from omnitrain.services.cleanup import cleanup_version_storage
from omnitrain.services.dataset_views import split_counts_for_dataset
from omnitrain.services.derived_items import delete_derived_items
from omnitrain.services.image_size import ImageDimensions, read_image_dimensions
from omnitrain.services.item_filter import TooManyMatches, build_item_filters, matching_item_ids
from omnitrain.services.snapshot import build_snapshot
from omnitrain.services.task_registry import class_annotation_type, get_task_descriptor, is_classification_task

logger = logging.getLogger(__name__)

# Rows per multi-row classify upsert; each row binds 3 parameters.
CLASSIFY_CHUNK = 2000

router = APIRouter(tags=["datasets"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Clips are far larger than images or audio. The upload route holds one in memory at a time (see `upload_items`).
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024
SORTS = ("newest", "oldest", "filename")
_background: set[asyncio.Task] = set()


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0


# -- Versions --------------------------------------------------------------------------------


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


# -- Listing ---------------------------------------------------------------------------------


@router.get("/projects/{project_id}/items", response_model=ItemListResponse)
async def list_items(
    project: ProjectDep,
    session: SessionDep,
    version_id: Annotated[uuid.UUID | None, Query(alias="versionId")] = None,
    split: str | None = None,
    class_id: Annotated[str | None, Query(alias="classId")] = None,
    search: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(alias="perPage", ge=1, le=1000)] = 30,
    sort: str | None = None,
    origin: Annotated[Literal["all", "original", "augmented"], Query()] = "all",
    labeled_filter: Annotated[bool | None, Query(alias="labeled")] = None,
) -> ItemListResponse:
    """Paginated pool items for the draft (default) or a given version.

    `origin` narrows a snapshot to its real items or to its augmented copies (the draft has no copies).
    `labeled` keeps only items that do (true) or do not (false) carry the task's ground-truth annotation
    type, which `classId=unassigned` can't express for tasks whose ground truth isn't a class.
    """
    if version_id is not None:
        version = await session.get(DatasetVersion, version_id)
        if version is None or version.dataset_id != project.id:
            raise HTTPException(404, "Version not found")
    else:
        version = (
            await session.execute(
                sa.select(DatasetVersion).where(
                    DatasetVersion.dataset_id == project.id, DatasetVersion.version_tag.is_(None)
                )
            )
        ).scalar_one_or_none()
        if version is None:
            raise HTTPException(404, "Version not found")

    # "Labeled" means "carries this task own ground-truth annotation type": classification for most
    # tasks, but text_sequence for captioning and ASR.
    ground_truth = get_task_descriptor(project.task).annotation.type
    # The annotation type whose rows carry a class: one `classification` per item, or one `bounding_box` per box.
    class_type = class_annotation_type(project.task) or "classification"
    vi = DatasetVersionItem

    order = {
        "oldest": (DatasetItem.created_at.asc(), DatasetItem.id.asc()),
        "filename": (DatasetItem.external_id.asc(), DatasetItem.id.asc()),
    }.get(sort or "newest", (DatasetItem.created_at.desc(), DatasetItem.id.desc()))

    try:
        filters = build_item_filters(
            version_id=version.id,
            dataset_id=project.id,
            ground_truth=ground_truth,
            class_type=class_type,
            origin=origin,
            split=split,
            class_id=class_id,
            search=search,
            labeled=labeled_filter,
        )
    except ValueError:
        raise HTTPException(400, "classId must be a class id or 'unassigned'") from None
    base, raw = filters.base, filters.raw

    total = (await session.execute(sa.select(sa.func.count()).select_from(vi).where(*raw))).scalar_one()
    labeled = (
        await session.execute(
            sa.select(sa.func.count())
            .select_from(vi)
            .where(
                *raw,
                sa.exists().where(Annotation.item_id == vi.item_id, Annotation.annotation_type == ground_truth),
            )
        )
    ).scalar_one()
    members = (
        await session.execute(
            sa.select(vi.item_id, vi.split_type)
            .join(DatasetItem, DatasetItem.id == vi.item_id)
            .where(*raw)
            .order_by(*order)
            .limit(per_page)
            .offset((page - 1) * per_page)
        )
    ).all()
    base_total = (await session.execute(sa.select(sa.func.count()).select_from(vi).where(*base))).scalar_one()
    class_rows = (
        await session.execute(
            sa.select(Annotation.class_id, sa.func.count(sa.distinct(vi.item_id)))
            .select_from(vi)
            .join(
                Annotation,
                sa.and_(Annotation.item_id == vi.item_id, Annotation.annotation_type == class_type),
            )
            .where(*base)
            .group_by(Annotation.class_id)
        )
    ).all()
    classified = (
        await session.execute(
            sa.select(sa.func.count(sa.distinct(vi.item_id)))
            .select_from(vi)
            .join(
                Annotation,
                sa.and_(Annotation.item_id == vi.item_id, Annotation.annotation_type == class_type),
            )
            .where(*base)
        )
    ).scalar_one()

    items: list[ItemOut] = []
    if members:
        ids = [m[0] for m in members]
        rows = {
            i.id: i for i in (await session.execute(sa.select(DatasetItem).where(DatasetItem.id.in_(ids)))).scalars()
        }

        def by_item(model):
            return session.execute(sa.select(model).where(model.item_id.in_(ids)))

        text = {r.item_id: r for r in (await by_item(TextFeatures)).scalars()}
        vision = {r.item_id: r for r in (await by_item(VisionFeatures)).scalars()}
        audio = {r.item_id: r for r in (await by_item(AudioFeatures)).scalars()}
        video = {r.item_id: r for r in (await by_item(VideoFeatures)).scalars()}
        tabular = {r.item_id: r for r in (await by_item(TabularFeatures)).scalars()}
        anns: dict[uuid.UUID, list[Annotation]] = {}
        for a in (
            await session.execute(
                sa.select(Annotation).where(Annotation.item_id.in_(ids)).order_by(Annotation.created_at, Annotation.id)
            )
        ).scalars():
            anns.setdefault(a.item_id, []).append(a)
        source_ids = {i.source_item_id for i in rows.values() if i.source_item_id is not None}
        source_names: dict[uuid.UUID, str | None] = {}
        if source_ids:
            source_rows = await session.execute(
                sa.select(DatasetItem.id, DatasetItem.external_id).where(DatasetItem.id.in_(source_ids))
            )
            source_names = {item_id: name for item_id, name in source_rows.all()}

        for item_id, split_type in members:
            item = rows.get(item_id)
            if item is None:
                continue
            items.append(
                ItemOut(
                    **ItemRow.model_validate(item).model_dump(),
                    text_features=TextFeaturesOut.model_validate(text[item_id]) if item_id in text else None,
                    vision_features=VisionFeaturesOut.model_validate(vision[item_id]) if item_id in vision else None,
                    audio_features=AudioFeaturesOut.model_validate(audio[item_id]) if item_id in audio else None,
                    video_features=VideoFeaturesOut.model_validate(video[item_id]) if item_id in video else None,
                    tabular_features=TabularFeaturesOut.model_validate(tabular[item_id])
                    if item_id in tabular
                    else None,
                    annotations=[AnnotationOut.model_validate(a) for a in anns.get(item_id, [])],
                    split_type=split_type,
                    download_url=storage.get_download_url(C.BUCKET_DATASETS, item.storage_url, 3600)
                    if item.storage_url
                    else None,
                    thumbnail_url=f"/api/items/{item_id}/thumbnail?size={thumbnails.LIST_SIZE}"
                    if thumbnails.is_thumbnailable(item.storage_url)
                    else None,
                    source_item_id=item.source_item_id,
                    source_external_id=source_names.get(item.source_item_id) if item.source_item_id else None,
                    preprocessing=item.preprocessing,
                    augmentation=item.augmentation,
                )
            )

    return ItemListResponse(
        items=items,
        total=total,
        labeled_count=labeled,
        class_counts=[ClassCount(class_id=cid, count=n) for cid, n in class_rows if cid is not None],
        unassigned_count=base_total - classified,
        page=page,
        per_page=per_page,
    )


@router.get("/items/{item_id}/thumbnail", status_code=307, response_class=Response)
async def get_item_thumbnail(item: ItemDep, size: Annotated[int, Query()] = thumbnails.LIST_SIZE) -> RedirectResponse:
    """A cached JPEG thumbnail of an image item (a poster frame for a video), made on first request.

    Redirects to object storage.
    """
    if size not in storage.THUMBNAIL_SIZES:
        raise HTTPException(422, f"size must be one of {', '.join(str(s) for s in storage.THUMBNAIL_SIZES)}")
    if not thumbnails.is_thumbnailable(item.storage_url):
        raise HTTPException(404, "This item has no thumbnail")
    assert item.storage_url is not None
    loop = asyncio.get_running_loop()
    try:
        key = await loop.run_in_executor(None, thumbnails.ensure_thumbnail, item.storage_url, size)
    except Exception:
        # An undecodable or oversized image: the client falls back to the full-size download.
        logger.warning("Could not make a thumbnail for item %s", item.id, exc_info=True)
        raise HTTPException(422, "Could not make a thumbnail for this item") from None
    url = storage.get_download_url(C.BUCKET_DATASETS, key, 3600)
    # Cached for less than the presigned URL lives, so the browser never replays an expired redirect.
    return RedirectResponse(url, status_code=307, headers={"Cache-Control": "private, max-age=1800"})


# -- Adding items ----------------------------------------------------------------------------


@router.post("/projects/{project_id}/items", response_model=CreateItemsResponse)
async def create_items(body: CreateItemsBody, draft: DraftDep, session: SessionDep) -> CreateItemsResponse:
    """Add inline (text/tabular) items to the pool and draft, each with a split assignment."""
    project_id = draft.project.id
    class_ids = {a.class_id for i in body.items for a in (i.annotations or []) if a.class_id is not None}
    if class_ids and not svc.task_uses_label_classes(draft.project.task):
        raise HTTPException(400, svc.NO_LABEL_CLASSES)
    for class_id in class_ids:
        if not await svc.class_in_dataset(session, class_id, project_id):
            raise HTTPException(400, f"Unknown label class: {class_id}")

    created: list[ItemRow] = []
    failed: list[ItemFailure] = []
    try:
        # Fast path: the whole batch in one savepoint with batched inserts.
        async with session.begin_nested():
            rows = await svc.create_items_bulk(session, project_id, draft.draft.id, body.items)
        created = [ItemRow.model_validate(r) for r in rows]
    except Exception:
        # Something in the batch violated a constraint. Redo it one item at a time, each in its own
        # savepoint (an item, its features, annotations and membership are one unit), so a bad item
        # fails alone instead of failing the batch.
        logger.info("Bulk item insert failed, retrying item by item", exc_info=True)
        for index, data in enumerate(body.items):
            try:
                async with session.begin_nested():
                    item = await svc.create_item(session, project_id, draft.draft.id, data)
                created.append(ItemRow.model_validate(item))
            except Exception as e:
                logger.error("Failed to create item at index %d", index, exc_info=e)
                failed.append(ItemFailure(index=index, message=svc.safe_item_error(e)))
    await session.commit()
    return CreateItemsResponse(created=created, failed=failed)


class _UploadBox(BaseModel):
    """One box of a file in an upload: a class, and the box in image fractions (validated like any annotation)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_id: uuid.UUID = Field(alias="classId")
    x: float
    y: float
    w: float
    h: float


async def _parse_upload_boxes(
    raw: str | None, file_count: int, allowed: bool, project_id: uuid.UUID, session: AsyncSession
) -> list[list[tuple[uuid.UUID | None, dict[str, Any]]] | None]:
    """`boxes` as one entry per file: None (leave unlabelled) or the (class, label) rows to create, where a file with
    nothing in it gets the "no objects" marker. A malformed `boxes` rejects the whole upload, before anything is stored.
    """
    if raw is None:
        return [None] * file_count
    if not allowed:
        raise HTTPException(400, "Boxes only apply to an object-detection project")
    try:
        parsed = TypeAdapter(list[list[_UploadBox] | None]).validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, f"Invalid boxes: {e.errors()[0]['msg']}") from None
    if len(parsed) != file_count:
        raise HTTPException(422, f"boxes needs one entry per file: got {len(parsed)} for {file_count} files")

    class_ids = {b.class_id for entry in parsed for b in entry or []}
    for class_id in class_ids:
        if not await svc.class_in_dataset(session, class_id, project_id):
            raise HTTPException(400, f"Unknown label class: {class_id}")

    result: list[list[tuple[uuid.UUID | None, dict[str, Any]]] | None] = []
    for entry in parsed:
        if entry is None:
            result.append(None)
            continue
        rows: list[tuple[uuid.UUID | None, dict[str, Any]]] = []
        for b in entry:
            label = {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
            try:
                validate_box_annotation(b.class_id, label)
            except ValueError as e:
                raise HTTPException(422, str(e)) from None
            rows.append((b.class_id, label))
        result.append(rows or [(None, {"background": True})])
    return result


@dataclass
class _Pooled:
    """What one uploaded file became before the database is touched: stored, or rejected with a reason."""

    upload: storage.PoolUpload | None = None
    reason: str | None = None
    image: ImageDimensions | None = None
    video: video_media.VideoInfo | None = None


@router.post("/projects/{project_id}/upload", response_model=UploadResponse)
async def upload_items(
    draft: DraftDep,
    session: SessionDep,
    files: Annotated[list[UploadFile], File()],
    split: Annotated[str, Form()],
    class_id: Annotated[uuid.UUID | None, Form(alias="classId")] = None,
    boxes: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """Upload files (vision/audio/video) into the pool and draft with one split (and optionally one class).

    For an object-detection project, `boxes` labels the files instead of a class: a JSON list with one entry per file,
    in order, each either null (leave it unlabelled), `[]` (labelled, nothing in it) or a list of
    `{classId, x, y, w, h}` boxes (see services/boxes.py). As with `classId`, only a file that is new to the pool is
    labelled; one already there keeps whatever labels it has.
    """
    if split not in svc.SPLITS:
        raise HTTPException(422, f"split must be one of: {', '.join(svc.SPLITS)}")
    project_id = draft.project.id
    labels_with_boxes = get_task_descriptor(draft.project.task).annotation.type == "bounding_box"
    if class_id is not None and not svc.task_uses_label_classes(draft.project.task):
        raise HTTPException(400, svc.NO_LABEL_CLASSES)
    if class_id is not None and labels_with_boxes:
        raise HTTPException(400, "A class belongs to one box, not to a whole image: send `boxes` instead of `classId`")
    if class_id is not None and not await svc.class_in_dataset(session, class_id, project_id):
        raise HTTPException(400, "Unknown label class")
    file_boxes = await _parse_upload_boxes(boxes, len(files), labels_with_boxes, project_id, session)
    dataset = await session.get(Dataset, project_id)
    if dataset is None:
        raise HTTPException(404, "Dataset not found")

    loop = asyncio.get_running_loop()
    is_video = dataset.modality == "video"
    limit, limit_label = (MAX_VIDEO_UPLOAD_BYTES, "200 MB") if is_video else (MAX_UPLOAD_BYTES, "50 MB")

    # Each file is read, checked and stored inside one gated step, so at most `gate` files are in memory at once:
    # a batch of clips must not be held whole. S3 uploads run OUTSIDE any transaction: slow network I/O must not
    # hold one open, and a leftover pool object after a later DB failure is a harmless no-op (content addressed,
    # so a retry reuses it).
    gate = asyncio.Semaphore(1 if is_video else 4)

    async def to_pool(f: UploadFile) -> _Pooled:
        async with gate:
            data = await f.read(limit + 1)
            if len(data) > limit:
                return _Pooled(reason=f"File is larger than the {limit_label} limit")
            ext = os.path.splitext(f.filename or "")[1]
            try:
                container = video_media.container_for(f.content_type, ext) if is_video else None
                info = await loop.run_in_executor(None, video_media.probe, data, ext, container) if is_video else None
                dims = (
                    read_image_dimensions(data)
                    if dataset.modality == "vision" and (f.content_type or "").startswith("image/")
                    else None
                )
                upload = await loop.run_in_executor(
                    None, storage.upload_to_pool, str(project_id), data, ext, f.content_type
                )
            except video_media.VideoError as e:
                return _Pooled(reason=str(e))
            except Exception:
                logger.error("Pool upload failed for %s", f.filename, exc_info=True)
                return _Pooled(reason="Failed to store file")
            return _Pooled(upload=upload, image=dims, video=info)

    pooled = await asyncio.gather(*(to_pool(f) for f in files))

    results: list[UploadResult] = []
    for index, (f, got) in enumerate(zip(files, pooled, strict=True)):
        if got.upload is None:
            results.append(UploadResult(status="rejected", reason=got.reason))
            continue
        upload = got.upload
        try:
            async with session.begin_nested():
                existing = (
                    await session.execute(
                        sa.select(DatasetItem).where(
                            DatasetItem.dataset_id == project_id,
                            DatasetItem.content_hash == upload.hash,
                            DatasetItem.source_item_id.is_(None),  # never dedup onto a derived item
                        )
                    )
                ).scalar_one_or_none()
                item = existing
                if item is None:
                    item = DatasetItem(
                        dataset_id=project_id, external_id=f.filename, storage_url=upload.key,
                        content_hash=upload.hash, byte_size=upload.byte_size,
                    )  # fmt: skip
                    session.add(item)
                    await session.flush()
                    if got.image:
                        session.add(
                            VisionFeatures(
                                item_id=item.id, width=got.image.width, height=got.image.height, channels=3,
                                image_format="png" if f.content_type == "image/png" else "jpeg",
                            )
                        )  # fmt: skip
                    if got.video:
                        v = got.video
                        session.add(
                            VideoFeatures(
                                item_id=item.id, width=v.width, height=v.height,
                                duration_seconds=Decimal(str(v.duration_seconds)), fps=Decimal(str(v.fps)),
                                frame_count=v.frame_count, container=v.container, codec=v.codec,
                            )
                        )  # fmt: skip
                    # Only label a freshly created item: a dedup hit is content already in the pool,
                    # possibly labeled differently, and a same-batch upload must not overwrite that.
                    if class_id is not None:
                        session.add(Annotation(item_id=item.id, annotation_type="classification", class_id=class_id))
                    for box_class_id, label in file_boxes[index] or []:
                        session.add(
                            Annotation(
                                item_id=item.id, annotation_type="bounding_box", class_id=box_class_id,
                                label_structured=label,
                            )
                        )  # fmt: skip
                    await session.flush()
                elif item.deleted_at is not None:
                    item.deleted_at = None  # re-uploading a soft-deleted item exact content restores it
                await session.execute(
                    pg_insert(DatasetVersionItem)
                    .values(version_id=draft.draft.id, item_id=item.id, split_type=split)
                    .on_conflict_do_nothing()
                )
            results.append(
                UploadResult(
                    status="fulfilled",
                    value=ItemRowWithDuplicate(
                        **ItemRow.model_validate(item).model_dump(), is_duplicate=upload.is_duplicate
                    ),
                )
            )
        except Exception as e:
            logger.error("Failed to record uploaded file %s", f.filename, exc_info=e)
            results.append(UploadResult(status="rejected", reason=svc.safe_item_error(e)))
    await session.commit()
    return UploadResponse(results=results)


# -- Deleting, splitting, labeling -----------------------------------------------------------


async def _selected_item_ids(
    body: ItemSelection, project: Project, session: AsyncSession, draft_id: uuid.UUID | None = None
) -> list[uuid.UUID]:
    """The explicit ids, or whatever the filter matches in the draft right now (bulk edits only touch the draft)."""
    if body.item_ids is not None:
        return body.item_ids
    assert body.filter is not None  # ItemSelection guarantees exactly one selector
    if draft_id is None:
        draft_id = (
            await session.execute(
                sa.select(DatasetVersion.id).where(
                    DatasetVersion.dataset_id == project.id, DatasetVersion.version_tag.is_(None)
                )
            )
        ).scalar_one_or_none()
        if draft_id is None:
            raise HTTPException(404, "Draft dataset not found")
    try:
        filters = build_item_filters(
            version_id=draft_id,
            dataset_id=project.id,
            ground_truth=get_task_descriptor(project.task).annotation.type,
            class_type=class_annotation_type(project.task) or "classification",
            split=body.filter.split,
            class_id=body.filter.class_id,
            search=body.filter.search,
            labeled=body.filter.labeled,
        )
        return await matching_item_ids(session, filters, body.exclude_item_ids)
    except ValueError:
        raise HTTPException(400, "classId must be a class id or 'unassigned'") from None
    except TooManyMatches as e:
        raise HTTPException(400, str(e)) from None


@router.delete("/items/{item_id}", status_code=204)
async def delete_item(item: ItemDep, session: SessionDep) -> None:
    if item.source_item_id is not None:
        raise HTTPException(400, "Derived items belong to a snapshot; delete the snapshot instead")
    await svc.delete_item_from_pool(session, item.id, item.dataset_id)
    await session.commit()


@router.delete("/projects/{project_id}/items", response_model=DeleteItemsResponse)
async def delete_items(body: DeleteItemsBody, project: ProjectDep, session: SessionDep) -> DeleteItemsResponse:
    item_ids = await _selected_item_ids(body, project, session)
    outcomes = [
        DeleteOutcome(item_id=item_id, outcome=await svc.delete_item_from_pool(session, item_id, project.id))
        for item_id in item_ids
    ]
    await session.commit()
    return DeleteItemsResponse(results=outcomes)


@router.patch("/projects/{project_id}/items/split", response_model=SetSplitResponse)
async def set_items_split(body: SetSplitBody, draft: DraftDep, session: SessionDep) -> SetSplitResponse:
    """Bulk-reassign the draft split of pool items."""
    item_ids = await _selected_item_ids(body, draft.project, session, draft.draft.id)
    updated: list[uuid.UUID] = []
    for chunk in chunked(item_ids):
        updated += (
            (
                await session.execute(
                    sa.update(DatasetVersionItem)
                    .where(DatasetVersionItem.version_id == draft.draft.id, DatasetVersionItem.item_id.in_(chunk))
                    .values(split_type=body.split)
                    .returning(DatasetVersionItem.item_id)
                )
            )
            .scalars()
            .all()
        )
    await session.commit()
    return SetSplitResponse(updated=updated)


@router.post("/projects/{project_id}/items/classify", response_model=ClassifyResponse)
async def classify_items(body: ClassifyBody, project: ProjectDep, session: SessionDep) -> ClassifyResponse:
    """Bulk-assign a label class: creates or updates each item classification annotation."""
    if not svc.task_uses_label_classes(project.task):
        raise HTTPException(400, svc.NO_LABEL_CLASSES)
    if get_task_descriptor(project.task).annotation.type == "bounding_box":
        # A class here is the label of one box, drawn on one image; there is no whole-image class to assign.
        raise HTTPException(400, "Boxes are labelled one at a time: draw a box on the item instead")
    if not await svc.class_in_dataset(session, body.class_id, project.id):
        raise HTTPException(400, "Unknown label class")
    item_ids = await _selected_item_ids(body, project, session)
    # Done in chunks of rows, never one statement per item and never one statement for everything: each
    # chunk is one multi-row upsert (3 bind parameters per row), well under asyncpg's 32767-argument cap,
    # and in its own savepoint so a failing chunk costs only its own items.
    owned: set[uuid.UUID] = set()
    for chunk in chunked(item_ids):
        owned.update(
            (
                await session.execute(
                    sa.select(DatasetItem.id).where(DatasetItem.id.in_(chunk), DatasetItem.dataset_id == project.id)
                )
            ).scalars()
        )
    valid = [i for i in dict.fromkeys(item_ids) if i in owned]  # de-duplicated: ON CONFLICT can't hit a row twice
    updated = 0
    failed = len(item_ids) - len(valid)
    for chunk in chunked(valid, CLASSIFY_CHUNK):
        try:
            async with session.begin_nested():
                # One statement instead of read-then-write: with the partial unique index on (item_id)
                # WHERE annotation_type = 'classification', two concurrent classify calls can no
                # longer both see "no existing annotation" and both insert.
                stmt = pg_insert(Annotation).values(
                    [{"item_id": i, "annotation_type": "classification", "class_id": body.class_id} for i in chunk]
                )
                await session.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[Annotation.item_id],
                        # A LITERAL predicate, never a bind parameter: after five executions Postgres
                        # switches a prepared statement to a generic plan, which cannot prove the
                        # predicate matches the partial index, so every later item would fail with
                        # "no unique or exclusion constraint matching the ON CONFLICT specification".
                        index_where=sa.text("annotation_type = 'classification'"),
                        set_={"class_id": stmt.excluded.class_id},
                    )
                )
            updated += len(chunk)
        except Exception:
            logger.warning("Classify failed for a chunk of %d items", len(chunk), exc_info=True)
            failed += len(chunk)
    await session.commit()
    return ClassifyResponse(updated=updated, failed=failed)


@router.post(
    "/projects/{project_id}/items/auto-split", response_model=AutoSplitResponse, response_model_exclude_none=True
)
async def auto_split_items(body: AutoSplitBody, draft: DraftDep, session: SessionDep) -> AutoSplitResponse:
    """Randomly reassign every draft item split by ratio (stratified by class for classification tasks)."""
    rows = (
        (
            await session.execute(
                sa.select(DatasetVersionItem.item_id).where(DatasetVersionItem.version_id == draft.draft.id)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return AutoSplitResponse(updated=0)

    r = body.ratios
    ratios = (
        {"train": r.train, "validation": r.validation, "test": r.test}
        if r
        else {"train": 80, "validation": 10, "test": 10}
    )
    if sum(ratios.values()) <= 0:
        raise HTTPException(400, "Ratios must sum to a positive number")
    # Stratify by label class when the task actually has classes: grouping an unlabeled or regression
    # dataset by a nonexistent class would just be one big group, i.e. the flat behavior.
    stratify = body.stratify if body.stratify is not None else is_classification_task(draft.project.task)

    # Chunked: `rows` is the whole draft, far past asyncpg's 32767-argument limit on a big dataset.
    class_of: dict[uuid.UUID, uuid.UUID | None] = {}
    for chunk in chunked(rows):
        annotated = await session.execute(
            sa.select(Annotation.item_id, Annotation.class_id)
            .where(Annotation.item_id.in_(chunk), Annotation.class_id.is_not(None))
            # An item with several boxes keeps one class (a later row overwrites an earlier one): its first
            # box's. Rows only ever collide within a chunk, since an item is in exactly one chunk.
            .order_by(Annotation.created_at.desc(), Annotation.id.desc())
        )
        class_of.update({item_id: class_id for item_id, class_id in annotated.all()})
    plan = svc.plan_split([(i, str(class_of.get(i, "unassigned"))) for i in rows], ratios, stratify)

    for split_type in svc.SPLITS:
        for chunk in chunked(plan.groups[split_type]):
            await session.execute(
                sa.update(DatasetVersionItem)
                .where(DatasetVersionItem.version_id == draft.draft.id, DatasetVersionItem.item_id.in_(chunk))
                .values(split_type=split_type)
            )
    await session.commit()
    return AutoSplitResponse(
        updated=len(rows),
        stratified=stratify,
        splits=AutoSplitCounts(**{s: len(plan.groups[s]) for s in svc.SPLITS}),
        warnings=plan.warnings,
    )


# -- Health ----------------------------------------------------------------------------------


@router.get("/projects/{project_id}/dataset/health", response_model=DatasetHealthResponse)
async def get_dataset_health(draft: DraftDep, session: SessionDep) -> DatasetHealthResponse:
    """Health / EDA report for the draft: what training would actually see."""
    project = draft.project
    dataset = await session.get(Dataset, project.id)
    modality = dataset.modality
    ground_truth = get_task_descriptor(project.task).annotation.type
    vi = DatasetVersionItem
    in_draft = vi.version_id == draft.draft.id

    def empty() -> DatasetHealth:
        return DatasetHealth(
            item_count=0, labeled_count=0, unlabeled_count=0, modality=modality, class_distribution=[],
            small_classes=[], duplicate_content_hashes=0, missing_content_hash=0,
            vision=None, audio=None, video=None, text=None, tabular=None,
        )  # fmt: skip

    item_count = (await session.execute(sa.select(sa.func.count()).select_from(vi).where(in_draft))).scalar_one()
    if item_count == 0:
        return DatasetHealthResponse(health=empty())

    labeled = (
        await session.execute(
            sa.select(sa.func.count(sa.distinct(Annotation.item_id)))
            .join(vi, Annotation.item_id == vi.item_id)
            .where(in_draft, Annotation.annotation_type == ground_truth)
        )
    ).scalar_one()

    distribution: list[HealthClassCount] = []
    class_type = class_annotation_type(project.task)
    if class_type is not None:
        rows = (
            await session.execute(
                sa.select(LabelClass.class_id, LabelClass.name, sa.func.count(sa.distinct(Annotation.item_id)))
                .join(Annotation, Annotation.class_id == LabelClass.class_id)
                .join(vi, Annotation.item_id == vi.item_id)
                .where(in_draft, Annotation.annotation_type == class_type)
                .group_by(LabelClass.class_id, LabelClass.name)
            )
        ).all()
        distribution = sorted(
            (HealthClassCount(class_id=c, name=n, count=k) for c, n, k in rows), key=lambda c: -c.count
        )
    small = [c for c in distribution if c.count < svc.MIN_ITEMS_PER_CLASS]

    # Pool-wide integrity check: the (dataset, content_hash) dedup should keep both of these at 0.
    live = (
        DatasetItem.dataset_id == project.id,
        DatasetItem.deleted_at.is_(None),
        DatasetItem.source_item_id.is_(None),  # augmented copies belong to snapshots, not the pool
    )
    dup_groups = (
        await session.execute(
            sa.select(DatasetItem.content_hash)
            .where(*live, DatasetItem.content_hash.is_not(None))
            .group_by(DatasetItem.content_hash)
            .having(sa.func.count() > 1)
        )
    ).all()
    missing_hash = (
        await session.execute(sa.select(sa.func.count()).where(*live, DatasetItem.content_hash.is_(None)))
    ).scalar_one()

    def mma(mn, mx, av, integer: bool = False) -> MinMaxAvg:
        avg = svc.js_round(_num(av)) if integer else _num(av)
        return MinMaxAvg(min=_num(mn), max=_num(mx), avg=avg)

    vision = audio = video = text = tabular = None
    if modality == "vision":
        agg = (
            await session.execute(
                sa.select(
                    sa.func.count(), sa.func.min(VisionFeatures.width), sa.func.max(VisionFeatures.width),
                    sa.func.avg(VisionFeatures.width), sa.func.min(VisionFeatures.height),
                    sa.func.max(VisionFeatures.height), sa.func.avg(VisionFeatures.height),
                )
                .join(vi, VisionFeatures.item_id == vi.item_id)
                .where(in_draft)
            )
        ).one()  # fmt: skip
        formats = (
            await session.execute(
                sa.select(VisionFeatures.image_format, sa.func.count())
                .join(vi, VisionFeatures.item_id == vi.item_id)
                .where(in_draft)
                .group_by(VisionFeatures.image_format)
            )
        ).all()
        vision = VisionHealth(
            count=agg[0], width=mma(agg[1], agg[2], agg[3], True), height=mma(agg[4], agg[5], agg[6], True),
            formats={(f or "unknown"): n for f, n in formats},
        )  # fmt: skip
    elif modality == "audio":
        agg = (
            await session.execute(
                sa.select(
                    sa.func.count(), sa.func.min(AudioFeatures.duration_seconds),
                    sa.func.max(AudioFeatures.duration_seconds), sa.func.avg(AudioFeatures.duration_seconds),
                )
                .join(vi, AudioFeatures.item_id == vi.item_id)
                .where(in_draft)
            )
        ).one()  # fmt: skip
        rates = (
            await session.execute(
                sa.select(AudioFeatures.sample_rate_hz, sa.func.count())
                .join(vi, AudioFeatures.item_id == vi.item_id)
                .where(in_draft)
                .group_by(AudioFeatures.sample_rate_hz)
            )
        ).all()
        audio = AudioHealth(
            count=agg[0], duration_seconds=mma(agg[1], agg[2], agg[3]), sample_rates={str(r): n for r, n in rates}
        )
    elif modality == "video":
        agg = (
            await session.execute(
                sa.select(
                    sa.func.count(), sa.func.min(VideoFeatures.duration_seconds),
                    sa.func.max(VideoFeatures.duration_seconds), sa.func.avg(VideoFeatures.duration_seconds),
                    sa.func.min(VideoFeatures.width), sa.func.max(VideoFeatures.width),
                    sa.func.avg(VideoFeatures.width),
                    sa.func.min(VideoFeatures.height), sa.func.max(VideoFeatures.height),
                    sa.func.avg(VideoFeatures.height), sa.func.min(VideoFeatures.fps), sa.func.max(VideoFeatures.fps),
                    sa.func.avg(VideoFeatures.fps),
                )
                .join(vi, VideoFeatures.item_id == vi.item_id)
                .where(in_draft)
            )
        ).one()  # fmt: skip
        containers = (
            await session.execute(
                sa.select(VideoFeatures.container, sa.func.count())
                .join(vi, VideoFeatures.item_id == vi.item_id)
                .where(in_draft)
                .group_by(VideoFeatures.container)
            )
        ).all()
        video = VideoHealth(
            count=agg[0], duration_seconds=mma(agg[1], agg[2], agg[3]), width=mma(agg[4], agg[5], agg[6], True),
            height=mma(agg[7], agg[8], agg[9], True), fps=mma(agg[10], agg[11], agg[12]),
            containers={(c or "unknown"): n for c, n in containers},
        )  # fmt: skip
    elif modality == "text":
        agg = (
            await session.execute(
                sa.select(
                    sa.func.count(), sa.func.min(TextFeatures.token_count), sa.func.max(TextFeatures.token_count),
                    sa.func.avg(TextFeatures.token_count), sa.func.count(TextFeatures.token_count),
                )
                .join(vi, TextFeatures.item_id == vi.item_id)
                .where(in_draft)
            )
        ).one()  # fmt: skip
        langs = (
            await session.execute(
                sa.select(TextFeatures.language_code, sa.func.count())
                .join(vi, TextFeatures.item_id == vi.item_id)
                .where(in_draft)
                .group_by(TextFeatures.language_code)
            )
        ).all()
        text = TextHealth(
            count=agg[0],
            token_count=mma(agg[1], agg[2], agg[3], True) if agg[4] > 0 else None,
            languages={(lang or "unknown"): n for lang, n in langs},
        )
    elif modality == "tabular":
        n = (
            await session.execute(
                sa.select(sa.func.count())
                .select_from(TabularFeatures)
                .join(vi, TabularFeatures.item_id == vi.item_id)
                .where(in_draft)
            )
        ).scalar_one()
        tabular = TabularHealth(count=n)

    return DatasetHealthResponse(
        health=DatasetHealth(
            item_count=item_count, labeled_count=labeled, unlabeled_count=item_count - labeled, modality=modality,
            class_distribution=distribution, small_classes=small, duplicate_content_hashes=len(dup_groups),
            missing_content_hash=missing_hash, vision=vision, audio=audio, video=video, text=text,
            tabular=tabular,
        )
    )  # fmt: skip


# -- Annotations -----------------------------------------------------------------------------


def _decimal(v: float | None) -> Decimal | None:
    return None if v is None else Decimal(str(v))


async def _check_new_box(session: AsyncSession, item_id: uuid.UUID, class_id: uuid.UUID | None, label: Any) -> None:
    """Keep an item's boxes and its "no objects" marker consistent for a new `bounding_box` annotation (its shape was
    already validated with the request body, see `AnnotationIn`).

    Drawing a box on an item marked as having no objects clears the marker (the marker's meaning is gone, and the
    caller would otherwise have to delete it first). Marking an item that already has boxes is refused: that would
    silently contradict them, so the boxes have to be deleted on purpose.
    """
    existing = (
        (
            await session.execute(
                sa.select(Annotation).where(Annotation.item_id == item_id, Annotation.annotation_type == "bounding_box")
            )
        )
        .scalars()
        .all()
    )
    if is_background(label):
        if any(not is_background(a.label_structured) for a in existing):
            raise HTTPException(409, "This item has boxes; delete them before marking it as having no objects")
        if existing:
            raise HTTPException(409, "This item is already marked as having no objects")
        return
    for marker in (a for a in existing if is_background(a.label_structured)):
        await session.delete(marker)


@router.post("/items/{item_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(body: CreateAnnotationBody, item: ItemDep, session: SessionDep) -> AnnotationResponse:
    if body.class_id and not await svc.class_in_dataset(session, body.class_id, item.dataset_id):
        raise HTTPException(400, "Unknown label class")
    if body.annotation_type == "bounding_box":
        await _check_new_box(session, item.id, body.class_id, body.label_structured)
    annotation = Annotation(
        item_id=item.id, annotator_id=body.annotator_id, annotation_type=body.annotation_type,
        class_id=body.class_id, label_text_sequence=body.label_text_sequence,
        label_structured=body.label_structured, confidence_score=_decimal(body.confidence_score),
    )  # fmt: skip
    session.add(annotation)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        # The partial unique index allows one classification label per item.
        raise HTTPException(409, "This item already has a classification annotation") from None
    await session.refresh(annotation)
    return AnnotationResponse(annotation=AnnotationOut.model_validate(annotation))


@router.get("/items/{item_id}/annotations", response_model=AnnotationListResponse)
async def list_annotations(item: ItemDep, session: SessionDep) -> AnnotationListResponse:
    rows = (
        (
            await session.execute(
                sa.select(Annotation)
                .where(Annotation.item_id == item.id)
                .order_by(Annotation.created_at, Annotation.id)
            )
        )
        .scalars()
        .all()
    )
    return AnnotationListResponse(annotations=[AnnotationOut.model_validate(a) for a in rows])


@router.patch("/annotations/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    body: UpdateAnnotationBody, annotation: AnnotationDep, session: SessionDep
) -> AnnotationResponse:
    """Re-label in place (avoids delete-then-create). Only the fields that were sent are changed."""
    dataset_id = (
        await session.execute(sa.select(DatasetItem.dataset_id).where(DatasetItem.id == annotation.item_id))
    ).scalar_one()
    if body.class_id and not await svc.class_in_dataset(session, body.class_id, dataset_id):
        raise HTTPException(400, "Unknown label class")
    if annotation.annotation_type == "bounding_box":
        # Validate what the row WILL hold: only the fields that were sent change, the rest stay as stored.
        try:
            validate_box_annotation(
                body.class_id or annotation.class_id,
                body.label_structured if body.label_structured is not None else annotation.label_structured,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
    for field in body.model_fields_set:
        value = getattr(body, field)
        if value is None:
            continue
        setattr(annotation, field, _decimal(value) if field == "confidence_score" else value)
    await session.commit()
    await session.refresh(annotation)
    return AnnotationResponse(annotation=AnnotationOut.model_validate(annotation))


@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(annotation: AnnotationDep, session: SessionDep) -> None:
    await session.delete(annotation)
    await session.commit()
