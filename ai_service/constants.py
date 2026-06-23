import json
from pathlib import Path
from types import SimpleNamespace

with open(Path(__file__).parent.parent / "schema" / "constants.json", "r", encoding="utf-8") as file:
    CONSTANTS = json.loads(file.read(), object_hook=lambda d: SimpleNamespace(**d))

# Bucket names
BUCKET_DATASETS = CONSTANTS.BUCKET_DATASETS
BUCKET_TRAINING = CONSTANTS.BUCKET_TRAINING
BUCKET_MODELS = CONSTANTS.BUCKET_MODELS

TRAINING_CONFIG_FILENAME = CONSTANTS.TRAINING_CONFIG_FILENAME
DATASET_VERSION_FILENAME = CONSTANTS.DATASET_VERSION_FILENAME
