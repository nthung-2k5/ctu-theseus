import pandas as pd

from theseus.services.predict import build_batch_result_frame, build_inference_output, parse_prediction_row


def test_category_with_idx2str_list():
    predictions = pd.DataFrame({"class_probabilities": [[0.1, 0.7, 0.2]]})
    idx2str = ["cat", "dog", "bird"]

    result = parse_prediction_row("class", "category", predictions, idx2str, threshold=0.0)

    assert result == {"dog": 0.7, "bird": 0.2, "cat": 0.1}


def test_category_threshold_filters_low_confidence():
    predictions = pd.DataFrame({"class_probabilities": [[0.1, 0.7, 0.2]]})
    idx2str = ["cat", "dog", "bird"]

    result = parse_prediction_row("class", "category", predictions, idx2str, threshold=0.5)

    assert result == {"dog": 0.7}


def test_category_with_dict_probabilities():
    predictions = pd.DataFrame({"class_probabilities": [{"cat": 0.3, "dog": 0.6}]})

    result = parse_prediction_row("class", "category", predictions, idx2str=None, threshold=0.0)

    assert result == {"dog": 0.6, "cat": 0.3}


def test_number_falls_back_to_predictions_column():
    predictions = pd.DataFrame({"target_predictions": [42.5]})

    result = parse_prediction_row("target", "number", predictions, idx2str=None, threshold=0.0)

    assert result == {"target": 42.5}


def test_missing_probability_column_returns_empty():
    predictions = pd.DataFrame({"other_column": [1]})

    result = parse_prediction_row("class", "category", predictions, idx2str=["a", "b"], threshold=0.0)

    assert result == {}


def test_results_sorted_descending_by_confidence():
    predictions = pd.DataFrame({"class_probabilities": [[0.2, 0.5, 0.3]]})
    idx2str = ["a", "b", "c"]

    result = parse_prediction_row("class", "category", predictions, idx2str, threshold=0.0)

    assert list(result.keys()) == ["b", "c", "a"]


# ──────────────────────────────────────────────────────────────────
# build_inference_output — the tagged-union shape /api/inference returns
# (InferenceResponseSchema.output in server/lib/schema.ts), distinct from
# parse_prediction_row above (kept only for the export golden-sample path).
# ──────────────────────────────────────────────────────────────────


def test_classification_from_idx2str_list():
    predictions = pd.DataFrame({"class_probabilities": [[0.1, 0.7, 0.2]]})
    idx2str = ["cat", "dog", "bird"]

    result = build_inference_output("class", "category", predictions, idx2str)

    assert result == {
        "kind": "classification",
        "feature": "class",
        "classes": [
            {"label": "dog", "confidence": 0.7},
            {"label": "bird", "confidence": 0.2},
            {"label": "cat", "confidence": 0.1},
        ],
    }


def test_classification_respects_top_k():
    predictions = pd.DataFrame({"class_probabilities": [[0.1, 0.7, 0.2]]})
    idx2str = ["cat", "dog", "bird"]

    result = build_inference_output("class", "category", predictions, idx2str, top_k=1)

    assert result["classes"] == [{"label": "dog", "confidence": 0.7}]


def test_classification_from_dict_probabilities():
    predictions = pd.DataFrame({"class_probabilities": [{"cat": 0.3, "dog": 0.6}]})

    result = build_inference_output("class", "category", predictions, idx2str=None)

    assert result == {
        "kind": "classification",
        "feature": "class",
        "classes": [
            {"label": "dog", "confidence": 0.6},
            {"label": "cat", "confidence": 0.3},
        ],
    }


def test_regression_returns_scalar_value_not_a_confidence():
    predictions = pd.DataFrame({"target_predictions": [52000.0]})

    result = build_inference_output("target", "number", predictions, idx2str=None)

    assert result == {"kind": "regression", "feature": "target", "value": 52000.0}


def test_generated_text_output():
    predictions = pd.DataFrame({"answer_predictions": ["Paris is the capital of France."]})

    result = build_inference_output("answer", "text", predictions, idx2str=None)

    assert result == {"kind": "text", "feature": "answer", "text": "Paris is the capital of France."}


def test_token_sequence_output_pairs_tokens_with_predicted_tags():
    predictions = pd.DataFrame({"tags_predictions": [["B-PER", "O", "B-LOC"]]})

    result = build_inference_output(
        "tags", "sequence", predictions, idx2str=None, input_tokens=["Alice", "visited", "Paris"]
    )

    assert result == {
        "kind": "tokens",
        "feature": "tags",
        "tokens": [
            {"token": "Alice", "tag": "B-PER"},
            {"token": "visited", "tag": "O"},
            {"token": "Paris", "tag": "B-LOC"},
        ],
    }


def test_token_sequence_output_without_input_tokens_still_returns_tags():
    predictions = pd.DataFrame({"tags_predictions": [["O", "B-LOC"]]})

    result = build_inference_output("tags", "sequence", predictions, idx2str=None, input_tokens=None)

    assert result == {
        "kind": "tokens",
        "feature": "tags",
        "tokens": [{"token": "", "tag": "O"}, {"token": "", "tag": "B-LOC"}],
    }


# ──────────────────────────────────────────────────────────────────
# build_batch_result_frame — batch inference's input + predictions concat
# ──────────────────────────────────────────────────────────────────


def test_batch_result_frame_concatenates_inputs_and_predictions_columns():
    input_df = pd.DataFrame({"age": [34, 52], "income": [52000, 61000]})
    predictions = pd.DataFrame({"class_predictions": ["approved", "denied"], "class_probability": [0.9, 0.6]})

    result = build_batch_result_frame(input_df, predictions)

    assert list(result.columns) == ["age", "income", "class_predictions", "class_probability"]
    assert result.iloc[0].to_dict() == {
        "age": 34,
        "income": 52000,
        "class_predictions": "approved",
        "class_probability": 0.9,
    }
    assert len(result) == 2


def test_batch_result_frame_realigns_a_non_default_input_index():
    # Simulates reading a CSV with an odd index (e.g. after a prior filter) —
    # concatenation must not join on that index and produce NaN-padded rows.
    input_df = pd.DataFrame({"age": [34, 52]}, index=[7, 12])
    predictions = pd.DataFrame({"class_predictions": ["approved", "denied"]})

    result = build_batch_result_frame(input_df, predictions)

    assert result["age"].tolist() == [34, 52]
    assert result["class_predictions"].tolist() == ["approved", "denied"]
    assert not result.isna().any().any()
