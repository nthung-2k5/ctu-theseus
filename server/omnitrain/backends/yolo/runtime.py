"""Where Ultralytics keeps its files, and the one-time setup that keeps it from touching anything else.

Left alone, Ultralytics writes a settings file under the user's config directory, phones home
(`sync`), may `pip install` missing packages, and drops downloaded weights and `runs/` in the current
directory. Everything is pointed under `TEMP_DIR/yolo` instead.

This module imports ultralytics only inside `configure()`, so importing it is free.
"""

import os
import sys
from pathlib import Path

from omnitrain.settings import get_settings

# Files a trained run uploads under its results prefix (and `load` reads back from the downloaded copy).
WEIGHTS_FILENAME = "best.pt"
# Class names in model index order plus what inference needs to know (image size, input/output names).
# Written from the trained model itself: it is the only source of truth for the class order.
METADATA_FILENAME = "names.json"


def cache_dir() -> Path:
    return get_settings().temp_dir / "yolo"


def configure() -> None:
    """Point Ultralytics at `cache_dir()` and switch off its telemetry. Safe to call repeatedly.

    The environment variables only take effect if ultralytics has not been imported yet, so this must run
    before the first import; the settings update afterwards is the part that always applies.
    """
    root = cache_dir()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "config"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    (root / "weights").mkdir(parents=True, exist_ok=True)

    from ultralytics import settings

    settings.update(
        {
            "weights_dir": str(root / "weights"),
            "runs_dir": str(root / "runs"),
            "datasets_dir": str(root / "datasets"),
            "sync": False,
        }
    )


def weights_ref(weights: str) -> str:
    """What to hand `YOLO(...)`. A `.yaml` architecture is resolved by name from inside the package; a `.pt`
    checkpoint gets an absolute path in the weights cache, because Ultralytics downloads a bare name into the
    current directory but downloads a full path to exactly that path."""
    if weights.endswith(".yaml"):
        return weights
    return str(cache_dir() / "weights" / weights)


def device() -> int | str:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def dataloader_workers() -> int:
    # Dataloader worker processes need the `spawn` start method on Windows, which this process cannot satisfy.
    return 0 if sys.platform == "win32" else min(8, os.cpu_count() or 1)
