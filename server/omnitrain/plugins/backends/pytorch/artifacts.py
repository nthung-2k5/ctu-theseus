"""Model artifacts: the conversions `PytorchBackend.convert` builds export bundles from.

Only ONNX is offered, and it is exported straight from the trained `nn.Module`:

  * An image model takes a batch of images as a float tensor `[batch, 3, size, size]` named `images` (already resized,
    cropped and normalised: see manifest.py) and returns raw class logits, which is what every generated client applies
    its own softmax to.
  * An audio model takes a batch of log-mel spectrograms as a float tensor `[batch, 1, n_mels, frames]` named
    `spectrograms` (already computed and standardised, see audio.py and manifest.py) and returns raw class logits.
    Every generated client computes the spectrogram from the recipe in the manifest.
  * A text model takes a batch of word ids as an int64 tensor `[batch, max_tokens]` named `tokens` (the text already
    split and looked up in the vocabulary, see text.py and manifest.py) and returns raw class logits.
  * A table model takes one float tensor `[batch]` per column, named by the column, as RAW values: filling a missing
    value and standardising are inside the graph (so the manifest carries no `numberNormalization`). It returns logits
    for a classifier, or the number in the target's own units for a regressor, in one tensor named by the label column.

This module imports neither torch nor torchvision at module level.
"""

import os
from typing import Any

from omnitrain.plugins.backends.base import ARTIFACT_FILENAMES, Artifact

ARTIFACTS: dict[str, Artifact] = {"onnx": Artifact("onnx", ARTIFACT_FILENAMES["onnx"])}

IMAGE_INPUT_NAME = "images"
IMAGE_OUTPUT_NAME = "logits"
AUDIO_INPUT_NAME = "spectrograms"
TEXT_INPUT_NAME = "tokens"


def convert(model: Any, artifact_id: str, workdir: str) -> str:
    """Export `model` (a `PytorchLoadedModel`) and return the path of the written file."""
    import torch

    from omnitrain.plugins.backends.pytorch import network

    artifact = ARTIFACTS[artifact_id]
    out_dir = os.path.join(workdir, artifact.id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, artifact.filename)

    # Exported from a CPU copy in eval mode, so a model that is serving on the GPU is left alone.
    net = model.net
    original_device = next(net.parameters()).device
    net.to("cpu").eval()
    # Traced with two rows, not one: PyTorch treats a size of 1 as special and would pin the batch axis to it.
    batch = torch.export.Dim("batch")
    if model.meta["inputKind"] == "tabular":
        columns = model.spec.columns
        output_name = model.meta["outputName"]
        wrapped: torch.nn.Module = network.NamedColumns(net, columns).eval()
        export_args: dict[str, Any] = {
            "args": (),
            "kwargs": {"inputs": {c: torch.zeros(2) for c in columns}},
            "input_names": list(columns),
            "dynamic_shapes": {"inputs": {c: {0: batch} for c in columns}},
        }
    elif model.meta["inputKind"] == "audio":
        output_name = IMAGE_OUTPUT_NAME
        wrapped = net
        export_args = {
            "args": (torch.zeros(2, 1, model.spec.n_mels, model.spec.frames),),
            "input_names": [AUDIO_INPUT_NAME],
            "dynamic_shapes": ({0: batch},),
        }
    elif model.meta["inputKind"] == "text":
        output_name = IMAGE_OUTPUT_NAME
        wrapped = net
        export_args = {
            "args": (torch.zeros(2, model.spec.max_tokens, dtype=torch.long),),
            "input_names": [TEXT_INPUT_NAME],
            "dynamic_shapes": ({0: batch},),
        }
    else:
        output_name = IMAGE_OUTPUT_NAME
        wrapped = net
        export_args = {
            "args": (torch.zeros(2, 3, model.spec.size, model.spec.size),),
            "input_names": [IMAGE_INPUT_NAME],
            "dynamic_shapes": ({0: batch},),
        }
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                f=path,
                dynamo=True,
                # One self-contained file: a bundle carries the model as a single file, and torch's default of
                # writing the weights to a `model.onnx.data` beside it would ship a graph with no weights.
                external_data=False,
                output_names=[output_name],
                **export_args,
            )
    finally:
        net.to(original_device)
    if not os.path.isfile(path):
        raise RuntimeError(f"torch did not write {artifact.filename} for artifact {artifact_id!r}")
    return path
