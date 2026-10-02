"""Artifact conversion: Ludwig's export_model takes a DIRECTORY, not a file path."""

import os
from types import SimpleNamespace

import pytest

from theseus.backends.ludwig import artifacts
from theseus.backends.ludwig.artifacts import ARTIFACTS, convert


class FakeLudwigModel:
    """Mirrors ludwig.utils.model_export.ModelExporter: makedirs(save_path), write model.<ext> inside it."""

    def __init__(self, write: bool = True):
        self.calls: list[tuple[str, str]] = []
        self.write = write

    def export_model(self, save_path: str, format: str = "safetensors") -> None:
        self.calls.append((save_path, format))
        os.makedirs(save_path, exist_ok=True)
        if self.write:
            ext = {"onnx": "model.onnx", "torch_export": "model.pt2"}[format]
            with open(os.path.join(save_path, ext), "wb") as f:
                f.write(b"weights")


def test_convert_passes_a_directory_and_returns_the_file_inside_it(tmp_path):
    artifact = ARTIFACTS["torch_export"]
    model = FakeLudwigModel()
    path = convert(model, "torch_export", str(tmp_path))

    ((save_path, ludwig_format),) = model.calls
    assert ludwig_format == "torch_export"  # this backend's format id happens to match Ludwig's own
    assert not save_path.endswith((".onnx", ".pt2"))  # never a file path: Ludwig would make it a directory
    assert path == os.path.join(save_path, artifact.filename)
    assert os.path.isfile(path) and open(path, "rb").read() == b"weights"


def test_convert_onnx_writes_the_file_itself_instead_of_ludwigs_exporter(tmp_path, monkeypatch):
    def fake_export(model, path):
        with open(path, "wb") as f:
            f.write(b"onnx")

    monkeypatch.setattr(artifacts, "_export_onnx", fake_export)
    model = FakeLudwigModel()
    path = convert(model, "onnx", str(tmp_path))

    assert model.calls == []  # LudwigModel.export_model(format="onnx") is what fails with 'inputs'
    assert path == os.path.join(str(tmp_path), "onnx", "model.onnx")


def test_convert_fails_clearly_when_ludwig_wrote_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="did not write model.pt2"):
        convert(FakeLudwigModel(write=False), "torch_export", str(tmp_path))


def test_an_unknown_artifact_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        convert(FakeLudwigModel(), "zip", str(tmp_path))


def _tiny_ecd_like(num_classes: int):
    """Same shape as ludwig's ECD as far as ONNX export is concerned: `forward(inputs, mask=None)` taking
    ONE dict, a `Dense` head, and a plain Python int (`used_tokens`) next to the tensors in its output."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    from ludwig.utils.torch_utils import Dense

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Dense(input_size=2, output_size=num_classes)

        def forward(self, inputs, mask=None):
            x = torch.stack([inputs["a"], inputs["b"]], dim=1)
            return {"y::logits": self.head(x), "y::last_hidden": x, "used_tokens": 4}

        def get_model_inputs(self):
            return {"a": torch.rand(2), "b": torch.rand(2)}

    return SimpleNamespace(model=Tiny())


@pytest.mark.parametrize("num_classes", [1, 3])
def test_export_onnx_handles_ecd_style_forward(tmp_path, num_classes):
    """Regression: torch reads a trailing dict in the args tuple as kwargs ('missing a required argument:
    inputs'), `used_tokens` is an int no ONNX node can return, and a multi-class Dense squeezes a dim of 3."""
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    path = str(tmp_path / "model.onnx")
    artifacts._export_onnx(_tiny_ecd_like(num_classes), path)

    session = ort.InferenceSession(path)
    assert [i.name for i in session.get_inputs()] == ["a", "b"]
    assert [o.name for o in session.get_outputs()] == ["y::logits"]  # the name the exported clients match on
    for batch in (1, 5):  # batch is dynamic
        feed = {n: np.random.rand(batch).astype(np.float32) for n in ("a", "b")}
        (logits,) = session.run(None, feed)
        assert logits.shape == ((batch,) if num_classes == 1 else (batch, num_classes))


def test_export_onnx_restores_dense_forward(tmp_path):
    pytest.importorskip("onnxscript")
    from ludwig.utils.torch_utils import Dense

    before = Dense.forward
    artifacts._export_onnx(_tiny_ecd_like(3), str(tmp_path / "model.onnx"))
    assert Dense.forward is before
