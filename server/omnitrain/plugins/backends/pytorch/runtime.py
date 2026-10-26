"""Where this backend keeps its files, and the device it runs on. Imports torch only inside functions."""

import os
import sys
from pathlib import Path

from omnitrain.settings import get_settings

# Files a trained run uploads under its results prefix (and `load` reads back from the downloaded copy).
WEIGHTS_FILENAME = "model.pt"
# Class names in the model's own index order plus the input spec it was trained with. Written from the trained model's
# own training setup: the only source of truth for the class order (never the label_classes table).
METADATA_FILENAME = "names.json"


def cache_dir() -> Path:
    return get_settings().temp_dir / "pytorch"


def configure() -> None:
    """Keep torchvision's downloaded pretrained weights under `cache_dir()` instead of the user's home directory."""
    import torch

    root = cache_dir() / "torch"
    root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(root))


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dataloader_workers() -> int:
    # Dataloader worker processes need the `spawn` start method on Windows, which this process cannot satisfy.
    return 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)
