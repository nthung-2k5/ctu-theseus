"""CTU OmniTrain backend: FastAPI API plus the training job runner in one process."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version() -> str:
    # The project is a uv "virtual" one (no build backend), so it is usually not installed and has no metadata;
    # the version is then read from the pyproject.toml that sits beside the package.
    try:
        return version("ctu-omnitrain-ai")
    except PackageNotFoundError:
        pass
    try:
        return tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8"))["project"][
            "version"
        ]
    except Exception:
        return "unknown"


# Informational (shown for a worker, quoted in errors). Compatibility between a server and a worker is decided by
# `omnitrain.schemas.worker.PROTOCOL_VERSION`, which is bumped on purpose.
__version__ = _version()
