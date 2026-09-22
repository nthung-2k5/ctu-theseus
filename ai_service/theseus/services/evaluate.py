"""Evaluation-report building blocks shared by every trainer backend.

A backend's own `evaluate()` (see `theseus/backends/ludwig/evaluate.py` for the reference
implementation) builds the bounded confusion-matrix / per-class-metrics report; these are the
parts of that job that don't depend on which framework trained the model: which split to
evaluate against, and picking the most informative misclassified rows out of a raw predictions
frame. `top_errors` assumes the framework-neutral `{feature}_predictions` / `{feature}_probability`
prediction-frame column convention (see `LoadedModel.predict`).

Class-index correctness: a backend's report must label `confusionMatrix`/`perClass` with class
NAMES, never raw indices — this sidesteps the whole "which index does the model vs. Postgres think
class N is" class of bug (see the `label_classes` table) by never handling a bare index at all.
"""

from typing import Any

import pandas as pd

# report.json is stored whole in Postgres (the run_evaluations table
# runEvaluations.report jsonb column) — cap it.
MAX_TOP_ERRORS = 500
MAX_CLASSES_FOR_MATRIX = 200


def pick_eval_split(df: pd.DataFrame, split_column: str) -> tuple[pd.DataFrame, str]:
    """test -> validation -> whole dataset. Mirrors the fallback ladder in jobs/export.py's
    `_build_golden_sample`."""
    if split_column in df.columns:
        test_rows = df[df[split_column] == "test"]
        if len(test_rows) > 0:
            return test_rows, "test"
        validation_rows = df[df[split_column] == "validation"]
        if len(validation_rows) > 0:
            return validation_rows, "validation"
    return df, "full"


def metric(feature_stats: dict[str, Any], key: str) -> float | None:
    value = feature_stats.get(key)
    return round(float(value), 4) if value is not None else None


def top_errors(
    eval_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    output_column: str,
    feature_name: str,
    item_id_column: str,
) -> list[dict[str, Any]]:
    """Misclassified rows, most-confident-wrong-answer first — a confident mistake is the most
    informative one to show a user — capped at MAX_TOP_ERRORS."""
    pred_col = f"{feature_name}_predictions"
    prob_col = f"{feature_name}_probability"
    if pred_col not in predictions.columns or item_id_column not in eval_rows.columns:
        return []

    actual = eval_rows[output_column].reset_index(drop=True)
    predicted = predictions[pred_col].reset_index(drop=True)
    item_ids = eval_rows[item_id_column].reset_index(drop=True)
    confidence = predictions[prob_col].reset_index(drop=True) if prob_col in predictions.columns else None

    errors: list[dict[str, Any]] = []
    for i in range(len(actual)):
        actual_value = actual.iloc[i]
        predicted_value = predicted.iloc[i]
        if pd.isna(actual_value) or str(actual_value) == str(predicted_value):
            continue
        errors.append(
            {
                "itemId": str(item_ids.iloc[i]),
                "actual": str(actual_value),
                "predicted": str(predicted_value),
                "confidence": round(float(confidence.iloc[i]), 4) if confidence is not None else None,
            }
        )

    errors.sort(key=lambda e: e["confidence"] or 0.0, reverse=True)
    return errors[:MAX_TOP_ERRORS]
