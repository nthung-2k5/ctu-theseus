import pandas as pd

from theseus.services.predict import build_batch_result_frame


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
