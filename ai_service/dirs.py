import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))

# Directories for API to save incoming files before sending to worker
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

# Temporary directory for uploaded files
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))

DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./datasets"))
