"""Annotations on an item: classification labels, bounding boxes, and the "no objects" marker."""

import uuid
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain.api.deps import AnnotationDep, ItemDep, SessionDep
from omnitrain.db.models import (
    Annotation,
    DatasetItem,
    TextFeatures,
)
from omnitrain.domain.boxes import is_background, validate_box_annotation
from omnitrain.domain.token_tags import tag_count_error, tag_shape_error
from omnitrain.schemas.datasets import (
    AnnotationListResponse,
    AnnotationOut,
    AnnotationResponse,
    CreateAnnotationBody,
    UpdateAnnotationBody,
)
from omnitrain.services import datasets as svc

router = APIRouter()


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


async def _check_tags_fit_text(session: AsyncSession, item_id: uuid.UUID, tags: list[str]) -> None:
    """One tag per word of the item's own text (`domain/token_tags.py`): a tag list that is off by one would pair every
    later word with the wrong tag, and training would learn from it without any sign of a problem."""
    text = (await session.execute(sa.select(TextFeatures.raw_text).where(TextFeatures.item_id == item_id))).scalar()
    if text is None:
        raise HTTPException(400, "Token tags belong on a text item")
    if problem := tag_count_error(tags, text):
        raise HTTPException(422, problem)


@router.post("/items/{item_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(body: CreateAnnotationBody, item: ItemDep, session: SessionDep) -> AnnotationResponse:
    if body.class_id and not await svc.class_in_dataset(session, body.class_id, item.dataset_id):
        raise HTTPException(400, "Unknown label class")
    if body.annotation_type == "bounding_box":
        await _check_new_box(session, item.id, body.class_id, body.label_structured)
    if body.annotation_type == "token_tags":
        await _check_tags_fit_text(session, item.id, body.label_structured)
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
    if annotation.annotation_type == "token_tags" and body.label_structured is not None:
        if problem := tag_shape_error(body.label_structured):
            raise HTTPException(422, problem)
        await _check_tags_fit_text(session, annotation.item_id, body.label_structured)
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
