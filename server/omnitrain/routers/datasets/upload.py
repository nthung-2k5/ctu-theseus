"""Adding items to the pool: from existing keys (JSON) or by uploading files."""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain.db.models import (
    Annotation,
    Dataset,
    DatasetItem,
    DatasetVersionItem,
    VideoFeatures,
    VisionFeatures,
)
from omnitrain.deps import DraftDep, SessionDep
from omnitrain.domain import video as video_media
from omnitrain.domain.boxes import validate_box_annotation
from omnitrain.domain.image_size import ImageDimensions, read_image_dimensions
from omnitrain.domain.task_registry import get_task_descriptor
from omnitrain.schemas.datasets import (
    CreateItemsBody,
    CreateItemsResponse,
    ItemFailure,
    ItemRow,
    ItemRowWithDuplicate,
    UploadResponse,
    UploadResult,
)
from omnitrain.services import datasets as svc
from omnitrain.services import storage

router = APIRouter()

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Clips are far larger than images or audio. The upload route holds one in memory at a time (see `upload_items`).
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024


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
    `{classId, x, y, w, h}` boxes (see domain/boxes.py). As with `classId`, only a file that is new to the pool is
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
