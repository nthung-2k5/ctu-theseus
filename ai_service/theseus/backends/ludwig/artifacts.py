"""Model artifacts: the actual Ludwig conversions `LudwigBackend.convert` builds bundles from.

Several export formats share one artifact (every devkit and app ships the ONNX model), so
conversion is keyed by artifact, not by format: it runs once per run and artifact and is reused
afterwards (see jobs/export.py).
"""

import os
from typing import Any

from theseus.backends.base import ARTIFACT_FILENAMES, Artifact

# Value for LudwigModel.export_model(format=...), keyed by Artifact.id. Both artifact ids Ludwig
# supports happen to double as their own Ludwig format string.
_LUDWIG_FORMAT: dict[str, str] = {"onnx": "onnx", "torch_export": "torch_export"}

ARTIFACTS: dict[str, Artifact] = {aid: Artifact(aid, filename) for aid, filename in ARTIFACT_FILENAMES.items()}


def convert(model: Any, artifact_id: str, workdir: str) -> str:
    """Export `model` (a loaded `ludwig.api.LudwigModel`, typed `Any` so importing this module
    never requires importing ludwig itself) and return the path of the written file.

    LudwigModel.export_model treats its path as a DIRECTORY and writes the artifact's filename
    inside it, so pass a directory and locate the file, never a file path.
    """
    artifact = ARTIFACTS[artifact_id]
    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    model.export_model(out_dir, format=_LUDWIG_FORMAT[artifact_id])
    path = os.path.join(out_dir, artifact.filename)
    if not os.path.isfile(path):
        raise RuntimeError(f"Ludwig did not write {artifact.filename} for artifact {artifact_id!r}")
    return path
