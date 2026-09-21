"""Decompile a trained run's Ludwig artifacts into a Theseus-owned preprocessing.json.

This is the piece that makes a bare ONNX export usable. Ludwig's ONNX export is the raw graph: no
resize/normalize, no tokenization, no idx-to-label decode. Two sources feed the manifest:

  * training_runs.ludwig_config (Postgres): the exact compiled config (feature names, types and
    columns), the symmetric counterpart to services/ludwig_config.py.
  * training_set_metadata.json (S3, written by Ludwig at train time): per-feature preprocessing
    parameters and, critically, idx2str for category outputs.

idx2str MUST come from the metadata file, NEVER from the label_classes table. Postgres has no idea
what index order Ludwig assigned a category classes, so guessing wrong here silently mislabels every
prediction a generated client makes.
"""

import asyncio
import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.db.models import TrainingRun
from theseus.services import storage

# Standard ImageNet normalization constants (RGB mean/std, 0-1 scale): public, framework-agnostic
# values used by torchvision and virtually every vision library. Ludwig standardize_image
# "imagenet1k" preset applies exactly these; an unrecognized preset name is left as the raw string
# in ludwigPreprocessing rather than guessed at.
IMAGE_STANDARDIZATION_PRESETS: dict[str, dict[str, list[float]]] = {
    "imagenet1k": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

_NUMBER_STAT_KEYS = ("mean", "std", "min", "max", "q1", "q2", "q3")


def find_metadata_key(keys: list[str]) -> str | None:
    return next((k for k in keys if k.endswith("training_set_metadata.json")), None)


def build_manifest(run_id: str, ludwig_config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: turn the compiled config plus the training-set metadata into the preprocessing manifest."""
    inputs = ludwig_config.get("input_features") or []
    if not inputs:
        raise ValueError(f"Run {run_id} has no compiled Ludwig config to decompile")

    out_inputs: list[dict[str, Any]] = []
    for f in inputs:
        feature_meta = (meta or {}).get(f["name"]) or {}
        preprocessing = feature_meta.get("preprocessing")
        entry: dict[str, Any] = {"name": f["name"], "type": f["type"], "column": f["column"]}
        if preprocessing is not None:
            entry["ludwigPreprocessing"] = preprocessing

        preset = (preprocessing or {}).get("standardize_image")
        if isinstance(preset, str) and preset in IMAGE_STANDARDIZATION_PRESETS:
            entry["imageNormalization"] = IMAGE_STANDARDIZATION_PRESETS[preset]

        # Ludwig fit_transform_params() writes mean/std (zscore) or min/max (minmax) onto the feature
        # metadata dict directly, as SIBLINGS of `preprocessing`. `normalization` itself (the
        # transform name) is inside `preprocessing`. The exported graph has no normalization built
        # in, so a tabular client must replicate this exact transform on raw values.
        normalization = (preprocessing or {}).get("normalization")
        if f["type"] == "number" and normalization:
            norm: dict[str, Any] = {"type": normalization}
            for key in _NUMBER_STAT_KEYS:
                if feature_meta.get(key) is not None:
                    norm[key] = feature_meta[key]
            entry["numberNormalization"] = norm
        out_inputs.append(entry)

    outputs: list[dict[str, Any]] = []
    for f in ludwig_config.get("output_features") or []:
        entry = {"name": f["name"], "type": f["type"], "column": f["column"]}
        idx2str = ((meta or {}).get(f["name"]) or {}).get("idx2str")
        if idx2str is not None:
            entry["classes"] = idx2str
        outputs.append(entry)

    return {"schemaVersion": 1, "runId": run_id, "inputs": out_inputs, "outputs": outputs}


async def extract_preprocessing(session: AsyncSession, run_id: uuid.UUID) -> dict[str, Any]:
    run = await session.get(TrainingRun, run_id)
    if run is None:
        raise ValueError(f"Run {run_id} not found")
    loop = asyncio.get_running_loop()
    keys = await loop.run_in_executor(
        None, storage.list_keys, C.BUCKET_TRAINING, storage.training_results_prefix(str(run_id))
    )
    meta_key = find_metadata_key(keys)
    meta = None
    if meta_key:
        raw = await loop.run_in_executor(None, storage.download_bytes, C.BUCKET_TRAINING, meta_key)
        meta = json.loads(raw)
    return build_manifest(str(run_id), run.ludwig_config or {}, meta)
