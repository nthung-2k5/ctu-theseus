"""Model artifacts: the conversions `VideoBackend.convert` builds export bundles from.

Only ONNX is offered. The graph takes one clip as a float tensor `[batch, 3, frames, height, width]` (already
sampled, resized and normalised: see manifest.py) and returns raw class logits, which is what every consumer
applies its own softmax to. This module imports neither torch nor torchvision at module level.
"""

import os
from typing import Any

from omnitrain.plugins.backends.base import ARTIFACT_FILENAMES, Artifact

ARTIFACTS: dict[str, Artifact] = {"onnx": Artifact("onnx", ARTIFACT_FILENAMES["onnx"])}

INPUT_NAME = "clips"
OUTPUT_NAME = "logits"


def convert(model: Any, artifact_id: str, workdir: str) -> str:
    """Export `model` (a `VideoLoadedModel`) and return the path of the written file."""
    import torch

    artifact = ARTIFACTS[artifact_id]
    spec = model.spec
    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, artifact.filename)

    # Exported from a CPU copy in eval mode, so a model that is serving on the GPU is left alone.
    net = model.net
    original_device = next(net.parameters()).device
    net.to("cpu").eval()
    try:
        sample = torch.zeros(1, 3, spec.num_frames, spec.size, spec.size)
        batch = torch.export.Dim("batch")
        with torch.no_grad():
            torch.onnx.export(
                net,
                (sample,),
                f=path,
                dynamo=True,
                # One self-contained file: a bundle carries the model as a single file, and torch's default of
                # writing the weights to a `model.onnx.data` beside it would ship a graph with no weights.
                external_data=False,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_shapes=({0: batch},),
            )
    finally:
        net.to(original_device)
    if not os.path.isfile(path):
        raise RuntimeError(f"torch did not write {artifact.filename} for artifact {artifact_id!r}")
    return path
