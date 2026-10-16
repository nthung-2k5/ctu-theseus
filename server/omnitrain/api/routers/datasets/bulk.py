"""Deleting, splitting and classifying items, singly or by selection."""

import logging
import uuid

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from omnitrain.api.deps import DraftDep, ItemDep, ProjectDep, SessionDep
from omnitrain.db.chunks import chunked
from omnitrain.db.models import (
    Annotation,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    Project,
)
from omnitrain.domain.task_registry import class_annotation_type, get_task_descriptor, is_classification_task
from omnitrain.schemas.datasets import (
    AutoSplitBody,
    AutoSplitCounts,
    AutoSplitResponse,
    ClassifyBody,
    ClassifyResponse,
    DeleteItemsBody,
    DeleteItemsResponse,
    DeleteOutcome,
    ItemSelection,
    SetSplitBody,
    SetSplitResponse,
)
from omnitrain.services import datasets as svc
from omnitrain.services.item_filter import TooManyMatches, build_item_filters, matching_item_ids

router = APIRouter()

logger = logging.getLogger(__name__)

# Rows per multi-row classify upsert; each row binds 3 parameters.
CLASSIFY_CHUNK = 2000


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
