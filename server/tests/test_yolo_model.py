from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from omnitrain.plugins.backends.base import LoadedModel
from omnitrain.plugins.backends.yolo import model as model_mod
from omnitrain.plugins.backends.yolo import runtime
from omnitrain.plugins.backends.yolo.model import YoloLoadedModel

META = {
    "schemaVersion": 1,
    "task": "classify",
    "imgsz": 224,
    "imageColumn": "image_path",
    "outputName": "class",
    "names": ["cat", "dog", "bird"],
}


class FakeYOLO:
    def __init__(self, probs_by_name):
        self.probs_by_name, self.calls = probs_by_name, []

    def predict(self, sources, **kwargs):
        self.calls.append((list(sources), kwargs))
        return [SimpleNamespace(probs=SimpleNamespace(data=np.array(self.probs_by_name[s]))) for s in sources]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(runtime, "device", lambda: "cpu")


def make_model(probs_by_name):
    return YoloLoadedModel(FakeYOLO(probs_by_name), META)


def test_is_a_loaded_model_with_labels_from_the_model_metadata():
    m = make_model({})
    assert isinstance(m, LoadedModel)
    assert m.input_columns == ["image_path"]
    assert (m.output.name, m.output.kind, m.output.labels) == ("class", "classification", ["cat", "dog", "bird"])


def test_predict_returns_the_framework_neutral_prediction_columns():
    m = make_model({"a.png": [0.1, 0.7, 0.2], "b.png": [0.6, 0.3, 0.1]})
    frame = m.predict(pd.DataFrame({"image_path": ["a.png", "b.png"]}))

    assert list(frame["class_predictions"]) == ["dog", "cat"]
    assert list(frame["class_probability"]) == pytest.approx([0.7, 0.6])
    assert frame["class_probabilities"].iloc[0] == pytest.approx([0.1, 0.7, 0.2])
    call_sources, kwargs = m.yolo.calls[0]
    assert call_sources == ["a.png", "b.png"]
    assert kwargs["imgsz"] == 224 and kwargs["device"] == "cpu" and kwargs["verbose"] is False


def test_to_output_is_the_classification_shape_sorted_by_confidence_and_cut_to_top_k():
    m = make_model({"a.png": [0.1, 0.7, 0.2]})
    out = m.to_output(m.predict(pd.DataFrame({"image_path": ["a.png"]})), top_k=2)

    assert out == {
        "kind": "classification",
        "feature": "class",
        "classes": [{"label": "dog", "confidence": 0.7}, {"label": "bird", "confidence": 0.2}],
    }


def test_golden_prediction_is_a_flat_sorted_dict_above_the_threshold():
    m = make_model({"a.png": [0.1, 0.7, 0.2]})
    frame = m.predict(pd.DataFrame({"image_path": ["a.png"]}))

    assert m.golden_prediction(frame) == {"dog": 0.7, "bird": 0.2, "cat": 0.1}
    assert list(m.golden_prediction(frame)) == ["dog", "bird", "cat"]
    assert m.golden_prediction(frame, threshold=0.15) == {"dog": 0.7, "bird": 0.2}


def test_s3_uris_are_fetched_to_a_local_file_but_local_paths_are_used_as_they_are(monkeypatch):
    fetched = []

    def fake_download(bucket, key, local_path):
        fetched.append((bucket, key))
        return local_path

    monkeypatch.setattr(model_mod.storage, "download_file", fake_download)
    m = make_model({})
    seen = {}

    def predict(sources, **kw):
        seen["sources"] = list(sources)
        return [SimpleNamespace(probs=SimpleNamespace(data=np.array([1.0, 0.0, 0.0]))) for _ in sources]

    m.yolo.predict = predict
    m.predict(pd.DataFrame({"image_path": ["s3://theseus-datasets/pool/x.png", "/tmp/local.jpg"]}))

    assert fetched == [("theseus-datasets", "pool/x.png")]
    assert seen["sources"][0].endswith("0.png")
    assert seen["sources"][1] == "/tmp/local.jpg"


def test_load_reads_names_and_weights_from_the_downloaded_run(monkeypatch, tmp_path):
    import json
    import sys

    loaded = []

    class FakeUltralytics:
        class YOLO:
            def __init__(self, path):
                loaded.append(path)

    monkeypatch.setattr(runtime, "configure", lambda: None)
    monkeypatch.setitem(sys.modules, "ultralytics", FakeUltralytics)
    (tmp_path / "names.json").write_text(json.dumps(META))

    result = model_mod.load(str(tmp_path))

    assert loaded == [str(tmp_path / "best.pt")]
    assert result.output.labels == ["cat", "dog", "bird"]
