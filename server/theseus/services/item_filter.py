"""The one definition of "which items match these filters", shared by the item listing and the bulk
mutations, so "select all N matching" can never act on a different set than the one the user was looking at."""

import uuid
from dataclasses import dataclass
from typing import Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.db.models import Annotation, DatasetItem, DatasetVersionItem
from theseus.services import datasets as svc

vi = DatasetVersionItem

# Bulk mutations resolved from a filter act on at most this many items per request. Each one is still
# a per-item delete/classify (a storage call, a savepoint), so a larger number belongs in a background
# job, not in a synchronous request.
MAX_FILTER_BULK = 5000


@dataclass
class ItemFilters:
    # Version/origin/split/search-scoped but deliberately WITHOUT the class and labeled filters: the
    # class-count breakdown groups over this, so every class count reflects the current split and
    # search whatever class is selected.
    base: list[sa.ColumnElement[bool]]
    # `base` plus the class and labeled filters: exactly the rows the listing shows.
    raw: list[sa.ColumnElement[bool]]


def build_item_filters(
    *,
    version_id: uuid.UUID,
    dataset_id: uuid.UUID,
    ground_truth: str,
    class_type: str = "classification",
    origin: Literal["all", "original", "augmented"] = "all",
    split: str | None = None,
    class_id: str | None = None,
    search: str | None = None,
    labeled: bool | None = None,
) -> ItemFilters:
    """Clauses over `dataset_version_items`. `ground_truth` is the task's annotation type: "labeled" means
    "carries that type", which is classification for most tasks but text_sequence for captioning and ASR.
    `class_type` is the annotation type whose rows carry the class: `classification` (one per item) or, for object
    detection, `bounding_box` (one per box, so "in class X" means "has a box of class X").

    Raises ValueError for a `class_id` that is neither a class id nor 'unassigned'.
    """

    def classification_of(cid: uuid.UUID | None = None) -> sa.Exists:
        clauses = [Annotation.item_id == vi.item_id, Annotation.annotation_type == class_type]
        if cid is not None:
            clauses.append(Annotation.class_id == cid)
        return sa.exists().where(*clauses)

    base: list[sa.ColumnElement[bool]] = [vi.version_id == version_id]
    if origin != "all":
        # "original" here means "not an augmented copy": a preprocessed replacement counts as
        # original too, since it still stands in for exactly one real item. correlate(vi) only:
        # the listing query joins DatasetItem itself, and auto-correlation would otherwise strip
        # it from this subquery's FROM.
        is_original = sa.exists().where(DatasetItem.id == vi.item_id, DatasetItem.augmentation.is_(None)).correlate(vi)
        base.append(is_original if origin == "original" else ~is_original)
    if split in svc.SPLITS:
        base.append(vi.split_type == split)
    if search:
        base.append(
            vi.item_id.in_(
                sa.select(DatasetItem.id).where(
                    DatasetItem.dataset_id == dataset_id, DatasetItem.external_id.icontains(search, autoescape=True)
                )
            )
        )

    raw = list(base)
    if class_id == "unassigned":
        raw.append(~classification_of())
    elif class_id:
        raw.append(classification_of(uuid.UUID(class_id)))  # ValueError if it isn't a uuid
    if labeled is not None:
        has_ground_truth = sa.exists().where(
            Annotation.item_id == vi.item_id, Annotation.annotation_type == ground_truth
        )
        raw.append(has_ground_truth if labeled else ~has_ground_truth)
    return ItemFilters(base=base, raw=raw)


class TooManyMatches(Exception):
    """A filter matched more items than a single bulk request may act on."""

    def __init__(self, limit: int = MAX_FILTER_BULK) -> None:
        super().__init__(f"More than {limit} items match; narrow the filter and try again")
        self.limit = limit


async def matching_item_ids(
    session: AsyncSession, filters: ItemFilters, exclude: list[uuid.UUID] | None = None
) -> list[uuid.UUID]:
    """Ids of the items a filter selects, minus `exclude`. Raises `TooManyMatches` past `MAX_FILTER_BULK`."""
    stmt = sa.select(vi.item_id).where(*filters.raw).order_by(vi.item_id).limit(MAX_FILTER_BULK + 1)
    if exclude:
        stmt = stmt.where(vi.item_id.not_in(exclude))
    ids = list((await session.execute(stmt)).scalars())
    if len(ids) > MAX_FILTER_BULK:
        raise TooManyMatches
    return ids
