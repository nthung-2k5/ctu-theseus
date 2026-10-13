"""The dataset health report: class balance, duplicates, and per-modality feature statistics."""

from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter

from omnitrain.db.models import (
    Annotation,
    AudioFeatures,
    Dataset,
    DatasetItem,
    DatasetVersionItem,
    LabelClass,
    TabularFeatures,
    TextFeatures,
    VideoFeatures,
    VisionFeatures,
)
from omnitrain.deps import DraftDep, SessionDep
from omnitrain.domain.task_registry import class_annotation_type, get_task_descriptor
from omnitrain.schemas.datasets import (
    AudioHealth,
    DatasetHealth,
    DatasetHealthResponse,
    HealthClassCount,
    MinMaxAvg,
    TabularHealth,
    TextHealth,
    VideoHealth,
    VisionHealth,
)
from omnitrain.services import datasets as svc

router = APIRouter()


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0


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
