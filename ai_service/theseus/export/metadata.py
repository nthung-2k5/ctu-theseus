"""Fetch a trained run's metadata file (if its backend needs one) and hand it, with the compiled
config, to the run's backend to build the preprocessing.json.

training_runs.config (Postgres) is the exact compiled config; the second input, when a backend
declares one (`TrainerBackend.metadata_filename`, e.g. Ludwig's training_set_metadata.json), is an
S3 object under the run's results prefix. Which parts of that combination matter, and how (e.g.
idx2str MUST come from the metadata file, never from the label_classes table), is entirely up to
the backend — see backends/ludwig/manifest.py for the reference implementation.
"""

import asyncio
import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.backends.registry import get_backend
from theseus.db.models import TrainingRun
from theseus.services import storage


def find_metadata_key(keys: list[str], suffix: str) -> str | None:
    return next((k for k in keys if k.endswith(suffix)), None)


async def extract_preprocessing(session: AsyncSession, run_id: uuid.UUID) -> dict[str, Any]:
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    backend = get_backend(run.backend)
    loop = asyncio.get_running_loop()

    meta = None
    if backend.metadata_filename:
        keys = await loop.run_in_executor(
            None, storage.list_keys, C.BUCKET_TRAINING, storage.training_results_prefix(str(run_id))
        )
        meta_key = find_metadata_key(keys, backend.metadata_filename)
        if meta_key:
            raw = await loop.run_in_executor(None, storage.download_bytes, C.BUCKET_TRAINING, meta_key)
            meta = json.loads(raw)

    return backend.preprocessing_manifest(str(run_id), run.config or {}, meta)
