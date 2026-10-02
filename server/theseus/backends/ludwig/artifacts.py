"""Model artifacts: the actual Ludwig conversions `LudwigBackend.convert` builds bundles from.

Several export formats share one artifact (every devkit and app ships the ONNX model), so
conversion is keyed by artifact, not by format: it runs once per run and artifact and is reused
afterwards (see jobs/export.py).
"""

import contextlib
import os
from collections.abc import Generator
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
    path = os.path.join(out_dir, artifact.filename)
    if artifact_id == "onnx":
        _export_onnx(model, path)
    else:
        model.export_model(out_dir, format=_LUDWIG_FORMAT[artifact_id])
    if not os.path.isfile(path):
        raise RuntimeError(f"Ludwig did not write {artifact.filename} for artifact {artifact_id!r}")
    return path


def _export_onnx(model: Any, path: str) -> None:
    """Write `model`'s ONNX graph to `path`, bypassing `LudwigModel.export_model(format="onnx")`.

    Ludwig's exporter calls `torch.onnx.export(module, (sample_input,), ...)`, and torch reads a
    dict at the end of that args tuple as **kwargs**, so `forward(inputs, mask=None)` is called
    without `inputs` ("missing a required argument: 'inputs'"). Ludwig also leaves ECD's plain-int
    `used_tokens` in the outputs, which the ONNX converter can't emit, and loses the output names
    the exported clients match on (`{feature}::logits`). So export a wrapper that returns only the
    tensor logits, passing the input dict as an explicit keyword.
    """
    import torch

    module = model.model
    module.eval()
    sample = module.get_model_inputs()
    input_names = list(sample)

    with torch.no_grad():
        outputs = module(dict(sample))
    output_names = [k for k, v in outputs.items() if k.endswith("::logits") and isinstance(v, torch.Tensor)]
    if not output_names:
        output_names = [k for k, v in outputs.items() if isinstance(v, torch.Tensor)]

    class _Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = module

        def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
            out = self.module(inputs)
            return tuple(out[name] for name in output_names)

    batch = torch.export.Dim("batch")
    with _squeeze_only_singleton_dim():
        torch.onnx.export(
            _Logits().eval(),
            args=(),
            kwargs={"inputs": sample},
            f=path,
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes={"inputs": {name: {0: batch} for name in input_names}},
        )


@contextlib.contextmanager
def _squeeze_only_singleton_dim() -> Generator[None]:
    """Make `ludwig.utils.torch_utils.Dense` skip its `squeeze(dim=-1)` unless that dim is 1.

    `Dense.forward` squeezes the last dim unconditionally. PyTorch treats that as a no-op for a
    multi-class head, but the dynamo ONNX exporter emits an ONNX `Squeeze` for it, which is an
    error on a dim of size != 1 (onnxruntime refuses to load the model: "Dimension of input 1 must
    be 1 instead of 2"). The replacement is the same function wherever the squeeze is valid, so
    restoring it afterwards is only tidiness, not correctness.
    """
    import torch
    from ludwig.utils.torch_utils import Dense

    original = Dense.forward

    def forward(self: Any, input: torch.Tensor) -> torch.Tensor:
        output = self.dense(input)
        return output.squeeze(-1) if output.shape[-1] == 1 else output

    Dense.forward = forward
    try:
        yield
    finally:
        Dense.forward = original
