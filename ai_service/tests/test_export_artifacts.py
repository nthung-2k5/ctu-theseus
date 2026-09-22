"""Artifact conversion: Ludwig's export_model takes a DIRECTORY, not a file path."""

import os

import pytest

from theseus.export.artifacts import ARTIFACTS, convert, get_artifact


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


@pytest.mark.parametrize("artifact_id", sorted(ARTIFACTS))
def test_convert_passes_a_directory_and_returns_the_file_inside_it(tmp_path, artifact_id):
    artifact = get_artifact(artifact_id)
    model = FakeLudwigModel()
    path = convert(model, artifact, str(tmp_path))

    ((save_path, ludwig_format),) = model.calls
    assert ludwig_format == artifact.ludwig_format
    assert not save_path.endswith((".onnx", ".pt2"))  # never a file path: Ludwig would make it a directory
    assert path == os.path.join(save_path, artifact.filename)
    assert os.path.isfile(path) and open(path, "rb").read() == b"weights"


def test_convert_fails_clearly_when_ludwig_wrote_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="did not write model.onnx"):
        convert(FakeLudwigModel(write=False), get_artifact("onnx"), str(tmp_path))


def test_an_unknown_artifact_is_an_error():
    with pytest.raises(ValueError, match="Unknown model artifact 'zip'"):
        get_artifact("zip")


def test_every_export_format_points_at_a_real_artifact():
    from theseus.export.registry import list_export_formats

    for fmt in list_export_formats():
        assert fmt.artifact in ARTIFACTS, fmt.id
