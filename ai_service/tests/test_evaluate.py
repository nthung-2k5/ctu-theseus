import pandas as pd

from theseus.services.evaluate import MAX_TOP_ERRORS, _pick_eval_split, build_evaluation_report

SPLIT_COL = "split"
ITEM_ID_COL = "_theseus_item_id"


class _FakeOutputFeature:
    def __init__(self, name, type_, column):
        self.name = name
        self.type = type_
        self.column = column


class _FakeConfigObj:
    def __init__(self, output_feature):
        self.output_features = [output_feature]


class _FakeModel:
    """Stands in for a loaded LudwigModel — only the surface
    build_evaluation_report actually touches (config_obj.output_features[0],
    training_set_metadata, evaluate())."""

    def __init__(self, output_feature, training_set_metadata, eval_stats, predictions):
        self.config_obj = _FakeConfigObj(output_feature)
        self.training_set_metadata = training_set_metadata
        self._eval_stats = eval_stats
        self._predictions = predictions

    def evaluate(self, dataset, collect_predictions, collect_overall_stats):
        return self._eval_stats, self._predictions, "results"


# ──────────────────────────────────────────────────────────────────
# _pick_eval_split — test -> validation -> full fallback ladder
# ──────────────────────────────────────────────────────────────────


def test_pick_eval_split_prefers_test_when_present():
    df = pd.DataFrame({SPLIT_COL: ["train", "test", "validation", "test"], "x": [1, 2, 3, 4]})
    rows, split = _pick_eval_split(df, SPLIT_COL)
    assert split == "test"
    assert len(rows) == 2


def test_pick_eval_split_falls_back_to_validation_when_test_empty():
    df = pd.DataFrame({SPLIT_COL: ["train", "validation", "train"], "x": [1, 2, 3]})
    rows, split = _pick_eval_split(df, SPLIT_COL)
    assert split == "validation"
    assert len(rows) == 1


def test_pick_eval_split_falls_back_to_full_when_no_split_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    rows, split = _pick_eval_split(df, SPLIT_COL)
    assert split == "full"
    assert len(rows) == 3


def test_pick_eval_split_falls_back_to_full_when_neither_test_nor_validation_present():
    df = pd.DataFrame({SPLIT_COL: ["train", "train"], "x": [1, 2]})
    rows, split = _pick_eval_split(df, SPLIT_COL)
    assert split == "full"
    assert len(rows) == 2


# ──────────────────────────────────────────────────────────────────
# build_evaluation_report — category (classification)
# ──────────────────────────────────────────────────────────────────


def test_category_report_labels_axes_from_idx2str_not_row_order():
    # idx2str is deliberately NOT alphabetical / insertion order, so a test
    # that used e.g. sorted class names instead of idx2str would still pass
    # by accident unless this ordering is actually respected.
    idx2str = ["dog", "cat", "bird"]
    output_feature = _FakeOutputFeature("class", "category", "class")
    df = pd.DataFrame(
        {
            SPLIT_COL: ["test", "test", "test"],
            "class": ["dog", "cat", "bird"],
            ITEM_ID_COL: ["item-1", "item-2", "item-3"],
        }
    )
    eval_stats = {
        "class": {
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "per_class_stats": {
                "dog": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "true_positives": 1, "false_negatives": 0},
                "cat": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "true_positives": 1, "false_negatives": 0},
                "bird": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "true_positives": 1, "false_negatives": 0},
            },
            "overall_stats": {"token_accuracy": 1.0, "avg_f1_score_macro": 1.0},
        }
    }
    predictions = pd.DataFrame({"class_predictions": ["dog", "cat", "bird"], "class_probability": [0.9, 0.8, 0.7]})
    model = _FakeModel(output_feature, {"class": {"idx2str": idx2str}}, eval_stats, predictions)

    result = build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL)
    assert result is not None
    report, _predictions_df = result

    assert report["idx2str"] == idx2str
    assert report["confusionMatrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert set(report["perClass"].keys()) == {"dog", "cat", "bird"}
    assert report["perClass"]["bird"]["support"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert report["topErrors"] == []


def test_category_report_top_errors_joins_item_id_and_sorts_by_confidence():
    idx2str = ["cat", "dog"]
    output_feature = _FakeOutputFeature("class", "category", "class")
    df = pd.DataFrame(
        {
            SPLIT_COL: ["test", "test", "test"],
            "class": ["cat", "dog", "cat"],
            ITEM_ID_COL: ["item-a", "item-b", "item-c"],
        }
    )
    eval_stats = {"class": {"overall_stats": {"token_accuracy": 0.33, "avg_f1_score_macro": 0.3}}}
    # Row 0: correct (cat==cat) — excluded. Row 1: wrong, low confidence.
    # Row 2: wrong, high confidence — should sort first.
    predictions = pd.DataFrame(
        {
            "class_predictions": ["cat", "cat", "dog"],
            "class_probability": [0.99, 0.55, 0.95],
        }
    )
    model = _FakeModel(output_feature, {"class": {"idx2str": idx2str}}, eval_stats, predictions)

    result = build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL)
    assert result is not None
    report, _ = result

    assert len(report["topErrors"]) == 2
    assert report["topErrors"][0] == {"itemId": "item-c", "actual": "cat", "predicted": "dog", "confidence": 0.95}
    assert report["topErrors"][1] == {"itemId": "item-b", "actual": "dog", "predicted": "cat", "confidence": 0.55}


def test_category_report_top_errors_capped_at_max():
    idx2str = ["a", "b"]
    output_feature = _FakeOutputFeature("class", "category", "class")
    n = MAX_TOP_ERRORS + 50
    df = pd.DataFrame(
        {
            SPLIT_COL: ["test"] * n,
            "class": ["a"] * n,
            ITEM_ID_COL: [f"item-{i}" for i in range(n)],
        }
    )
    eval_stats = {"class": {"overall_stats": {"token_accuracy": 0.0, "avg_f1_score_macro": 0.0}}}
    # Every row wrong, so there are more errors than the cap.
    predictions = pd.DataFrame({"class_predictions": ["b"] * n, "class_probability": [0.5] * n})
    model = _FakeModel(output_feature, {"class": {"idx2str": idx2str}}, eval_stats, predictions)

    result = build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL)
    assert result is not None
    report, _ = result
    assert len(report["topErrors"]) == MAX_TOP_ERRORS


def test_category_report_truncates_matrix_for_very_high_class_count():
    idx2str = [f"class_{i}" for i in range(250)]
    output_feature = _FakeOutputFeature("class", "category", "class")
    df = pd.DataFrame({SPLIT_COL: ["test"], "class": ["class_0"], ITEM_ID_COL: ["item-1"]})
    eval_stats = {
        "class": {
            "confusion_matrix": [[1]],
            "per_class_stats": {},
            "overall_stats": {"token_accuracy": 1.0, "avg_f1_score_macro": 1.0},
        }
    }
    predictions = pd.DataFrame({"class_predictions": ["class_0"], "class_probability": [1.0]})
    model = _FakeModel(output_feature, {"class": {"idx2str": idx2str}}, eval_stats, predictions)

    result = build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL)
    assert result is not None
    report, _ = result
    assert report["truncated"] is True
    assert "confusionMatrix" not in report
    assert "perClass" not in report


# ──────────────────────────────────────────────────────────────────
# build_evaluation_report — number (regression)
# ──────────────────────────────────────────────────────────────────


def test_regression_report_reads_mae_rmse_r2():
    output_feature = _FakeOutputFeature("target", "number", "target")
    df = pd.DataFrame({SPLIT_COL: ["test", "test"], "target": [10.0, 20.0], ITEM_ID_COL: ["item-1", "item-2"]})
    eval_stats = {"target": {"mean_absolute_error": 1.5, "root_mean_squared_error": 2.5, "r2": 0.9}}
    predictions = pd.DataFrame({"target_predictions": [11.0, 18.0]})
    model = _FakeModel(output_feature, {}, eval_stats, predictions)

    result = build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL)
    assert result is not None
    report, _ = result

    assert report["outputType"] == "number"
    assert report["overall"] == {"mae": 1.5, "rmse": 2.5, "r2": 0.9}
    assert "topErrors" not in report


# ──────────────────────────────────────────────────────────────────
# Empty dataset
# ──────────────────────────────────────────────────────────────────


def test_returns_none_for_empty_dataset():
    output_feature = _FakeOutputFeature("class", "category", "class")
    df = pd.DataFrame({SPLIT_COL: [], "class": [], ITEM_ID_COL: []})
    model = _FakeModel(output_feature, {}, {}, pd.DataFrame())

    assert build_evaluation_report(model, df, SPLIT_COL, ITEM_ID_COL) is None
