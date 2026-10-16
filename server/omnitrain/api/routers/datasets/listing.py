"""Listing and filtering the items of a project's pool, and item thumbnails."""

import asyncio
import logging
import uuid
from typing import Annotated, Literal

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, Response

from omnitrain.api.deps import ItemDep, ProjectDep, SessionDep
from omnitrain.core import constants as C
from omnitrain.db.models import (
    Annotation,
    AudioFeatures,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    TabularFeatures,
    TextFeatures,
    VideoFeatures,
    VisionFeatures,
)
from omnitrain.domain.task_registry import class_annotation_type, get_task_descriptor
from omnitrain.schemas.datasets import (
    AnnotationOut,
    AudioFeaturesOut,
    ClassCount,
    ItemListResponse,
    ItemOut,
    ItemRow,
    TabularFeaturesOut,
    TextFeaturesOut,
    VideoFeaturesOut,
    VisionFeaturesOut,
)
from omnitrain.services import storage, thumbnails
from omnitrain.services.item_filter import build_item_filters

logger = logging.getLogger(__name__)

router = APIRouter()


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
