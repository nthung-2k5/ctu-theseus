import numpy as np
import pandas as pd
import pytest

from omnitrain.backends.yolo import evaluate as evaluate_mod
from omnitrain.backends.yolo import runtime
from omnitrain.backends.yolo.model import YoloLoadedModel

LABELS = ["cat", "dog", "bird"]
META = {"task": "classify", "imgsz": 32, "imageColumn": "image_path", "outputName": "class", "names": LABELS}

# What the fake model "sees" in each image: path -> class probabilities in LABELS order.
PROBS = {
    "c1": [0.9, 0.05, 0.05],  # cat, right
    "c2": [0.2, 0.7, 0.1],  # cat, called dog
    "d1": [0.1, 0.8, 0.1],  # dog, right
    "d2": [0.3, 0.6, 0.1],  # dog, right
    "b1": [0.1, 0.8, 0.1],  # bird, called dog
}
ACTUAL = {"c1": "cat", "c2": "cat", "d1": "dog", "d2": "dog", "b1": "bird"}


class FakeYOLO:
    def predict(self, sources, **kwargs):
        from types import SimpleNamespace

        return [SimpleNamespace(probs=SimpleNamespace(data=np.array(PROBS[s]))) for s in sources]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(runtime, "device", lambda: "cpu")


def frame(split="test", ids=tuple(PROBS)):
    return pd.DataFrame(
        {
            "image_path": list(ids),
            "class": [ACTUAL[i] for i in ids],
            "split": [split] * len(ids),
            "_theseus_item_id": [f"item-{i}" for i in ids],
        }
    )


def evaluate(df, labels=LABELS):
    model = YoloLoadedModel(FakeYOLO(), {**META, "names": labels})
    return evaluate_mod.evaluate(model, df, "split", "_theseus_item_id")


def test_confusion_matrix_has_actual_on_rows_and_predicted_on_columns_in_model_label_order():
    result = evaluate(frame())
    assert result is not None
    report = result.report

    assert report["idx2str"] == LABELS
    #            pred: cat dog bird
    assert report["confusionMatrix"] == [
        [1, 1, 0],  # actual cat
        [0, 2, 0],  # actual dog
        [0, 1, 0],  # actual bird
    ]
    assert report["split"] == "test" and report["rowCount"] == 5 and report["outputType"] == "category"


def test_overall_and_per_class_statistics():
    report = evaluate(frame()).report  # type: ignore[union-attr]

    assert report["overall"]["accuracy"] == pytest.approx(0.6)
    cat, dog, bird = (report["perClass"][name] for name in LABELS)
    assert cat == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-4), "support": 2}
    assert dog == {"precision": pytest.approx(0.5), "recall": 1.0, "f1": pytest.approx(0.6667, abs=1e-4), "support": 2}
    assert bird == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert report["overall"]["macroF1"] == pytest.approx((0.6667 + 0.6667 + 0.0) / 3, abs=1e-3)


def test_top_errors_are_the_most_confident_mistakes_first():
    errors = evaluate(frame()).report["topErrors"]  # type: ignore[union-attr]
    assert [(e["itemId"], e["actual"], e["predicted"]) for e in errors] == [
        ("item-b1", "bird", "dog"),
        ("item-c2", "cat", "dog"),
    ]
    assert errors[0]["confidence"] == pytest.approx(0.8)


def test_predictions_download_carries_item_ids_and_actual_labels():
    predictions = evaluate(frame()).predictions  # type: ignore[union-attr]
    assert list(predictions["_theseus_item_id"]) == [f"item-{i}" for i in PROBS]
    assert list(predictions["class_actual"]) == [ACTUAL[i] for i in PROBS]
    assert {"class_predictions", "class_probabilities", "class_probability"} <= set(predictions.columns)


def test_evaluates_test_then_validation_then_everything():
    df = pd.concat([frame("train", ("c1", "d1")), frame("validation", ("c2",))], ignore_index=True)
    assert evaluate(df).report["split"] == "validation"  # type: ignore[union-attr]
    only_train = frame("train")
    assert evaluate(only_train).report["split"] == "full"  # type: ignore[union-attr]


def test_rows_with_unknown_classes_or_no_image_are_left_out_and_nothing_left_means_no_report():
    df = frame()
    df.loc[0, "class"] = "fish"
    df.loc[1, "image_path"] = None
    result = evaluate(df)
    assert result is not None and result.report["rowCount"] == 3

    assert evaluate(df.iloc[:0]) is None


def test_matrix_and_per_class_are_dropped_past_the_class_limit(monkeypatch):
    monkeypatch.setattr(evaluate_mod, "MAX_CLASSES_FOR_MATRIX", 2)
    report = evaluate(frame()).report  # type: ignore[union-attr]
    assert report["truncated"] is True
    assert "confusionMatrix" not in report and "perClass" not in report
    assert report["overall"]["accuracy"] == pytest.approx(0.6)


def test_a_class_never_seen_or_predicted_does_not_drag_macro_f1_down():
    labels = [*LABELS, "unused"]
    report = evaluate(frame(), labels).report  # type: ignore[union-attr]
    # Same three classes as before: the fourth has no support and no predictions.
    assert report["overall"]["macroF1"] == pytest.approx((0.6667 + 0.6667 + 0.0) / 3, abs=1e-3)
