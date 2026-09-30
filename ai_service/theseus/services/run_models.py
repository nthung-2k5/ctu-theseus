"""Putting a trained run's bring-your-own model back where it was trained.

A run that trained on a custom model saved a path to that model's files inside its compiled config and its
checkpoint (Ludwig re-reads the tokenizer and architecture from there when the model is loaded). The path
is deterministic (see services/custom_models.py) but the files are temporary, so a restart, a cache
eviction or a fresh worker means they must be recreated before the model is trained, loaded for inference,
or converted for export. This is that one step; every load site calls it, and it is a no-op for the
(overwhelmingly common) run that used a built-in model.
"""

import asyncio
import uuid

import sqlalchemy as sa

from theseus.db.base import get_sessionmaker
from theseus.db.models import CustomModel, TrainingRun
from theseus.services import custom_models


async def ensure_run_custom_model(run_id: uuid.UUID | str) -> None:
    try:
        rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id))
    except ValueError:
        # Not a run id, so it cannot have trained on a custom model. (Real run ids are always UUIDs, and
        # `resolve_backend` has already rejected an unknown one before the model cache gets here; this also
        # keeps the cache's own tests, which use made-up ids and no database, independent of the model tables.)
        return
    async with get_sessionmaker()() as s:
        row = (
            await s.execute(
                sa.select(CustomModel)
                .join(TrainingRun, TrainingRun.custom_model_id == CustomModel.id)
                .where(TrainingRun.id == rid)
            )
        ).scalar_one_or_none()
    if row is not None:
        await asyncio.to_thread(custom_models.ensure_local, row)
