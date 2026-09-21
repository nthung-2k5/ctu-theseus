"""Shared response-building helpers for dataset versions (used by the projects and datasets routers)."""

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.db.models import DatasetVersion, DatasetVersionItem
from theseus.schemas.projects import SplitCounts, VersionOut


async def split_counts_for_dataset(session: AsyncSession, dataset_id: uuid.UUID) -> dict[uuid.UUID, SplitCounts]:
    """Per-version split membership counts in ONE grouped query (at most three rows per version).

    Deliberately not a full membership load: the project route is hit on every navigation and a
    large pool has hundreds of thousands of membership rows, while every consumer only reads counts.
    """
    rows = (
        await session.execute(
            sa.select(DatasetVersionItem.version_id, DatasetVersionItem.split_type, sa.func.count())
            .join(DatasetVersion, DatasetVersion.id == DatasetVersionItem.version_id)
            .where(DatasetVersion.dataset_id == dataset_id)
            .group_by(DatasetVersionItem.version_id, DatasetVersionItem.split_type)
        )
    ).all()
    out: dict[uuid.UUID, SplitCounts] = {}
    for version_id, split, n in rows:
        out.setdefault(version_id, SplitCounts())
        setattr(out[version_id], split, n)
    return out


def version_with_counts(version: DatasetVersion, counts: dict[uuid.UUID, SplitCounts]) -> VersionOut:
    """Attach split counts and a live item total (the item_count column is only written at snapshot time,
    so it is null for the draft, which is exactly the version the UI shows a live count for)."""
    split_counts = counts.get(version.id) or SplitCounts()
    out = VersionOut.model_validate(version)
    out.split_counts = split_counts
    out.item_count = split_counts.train + split_counts.validation + split_counts.test
    return out
