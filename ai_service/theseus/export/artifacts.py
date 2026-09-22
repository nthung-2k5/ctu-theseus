"""Model artifacts: the actual Ludwig conversions that export formats build their bundles from.

Several formats share one artifact (every devkit and app ships the ONNX model), so conversion is
keyed by artifact, not by format: it runs once per run and artifact and is reused afterwards.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Artifact:
    id: str
    # File name Ludwig writes inside the export directory, and the S3 name under the run prefix.
    filename: str
    # Value for LudwigModel.export_model(format=...).
    ludwig_format: str


ARTIFACTS: dict[str, Artifact] = {
    a.id: a
    for a in (
        Artifact("onnx", "model.onnx", "onnx"),
        Artifact("torch_export", "model.pt2", "torch_export"),
    )
}


def get_artifact(artifact_id: str) -> Artifact:
    try:
        return ARTIFACTS[artifact_id]
    except KeyError:
        raise ValueError(f"Unknown model artifact {artifact_id!r}") from None


def convert(model: Any, artifact: Artifact, workdir: str) -> str:
    """Export `model` and return the path of the written file.

    LudwigModel.export_model treats its path as a DIRECTORY and writes `artifact.filename` inside
    it, so pass a directory and locate the file, never a file path.
    """
    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    model.export_model(out_dir, format=artifact.ludwig_format)
    path = os.path.join(out_dir, artifact.filename)
    if not os.path.isfile(path):
        raise RuntimeError(f"Ludwig did not write {artifact.filename} for artifact {artifact.id!r}")
    return path
