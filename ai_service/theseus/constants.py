"""Shared constants, loaded from the repo-level schema/constants.json (also read by web/ and apphost.mts)."""

import json
import os
from pathlib import Path


def _schema_dir() -> Path:
    if env := os.environ.get("SCHEMA_DIR"):
        return Path(env)
    # Inside the container the AppHost bind-mounts ./schema at /schema.
    if Path("/schema/constants.json").exists():
        return Path("/schema")
    # ai_service/theseus/constants.py -> the repo root is two levels above the package.
    return Path(__file__).resolve().parents[2] / "schema"


SCHEMA_DIR = _schema_dir()

with open(SCHEMA_DIR / "constants.json", encoding="utf-8") as _f:
    _RAW = json.load(_f)

SPLIT_COLUMN_NAME: str = _RAW["SPLIT_COLUMN_NAME"]
SPLIT_INDEX_COLUMN_NAME: str = _RAW["SPLIT_INDEX_COLUMN_NAME"]
ITEM_ID_COLUMN_NAME: str = _RAW["ITEM_ID_COLUMN_NAME"]
BUCKET_DATASETS: str = _RAW["BUCKET_DATASETS"]
BUCKET_TRAINING: str = _RAW["BUCKET_TRAINING"]
BUCKET_MODELS: str = _RAW["BUCKET_MODELS"]
# Inference uploads that must outlive the request (replaces the NATS object store).
TRAINING_CONFIG_FILENAME: str = _RAW["TRAINING_CONFIG_FILENAME"]
DATASET_VERSION_FILENAME: str = _RAW["DATASET_VERSION_FILENAME"]
IMAGE_PATH_COLUMN_NAME: str = _RAW["DATASET"]["IMAGE_PATH_COLUMN_NAME"]
CLASS_COLUMN_NAME: str = _RAW["DATASET"]["CLASS_COLUMN_NAME"]
