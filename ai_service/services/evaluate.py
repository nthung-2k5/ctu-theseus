"""Evaluation reporting for a just-trained model.

Builds a bounded confusion-matrix / per-class-metrics report plus a capped
list of misclassified rows, on top of Ludwig's own `ConfusionMatrix`
(`ludwig.utils.eval_utils`, the same class `category_feature.py`'s
`calculate_overall_stats` uses internally) rather than recomputing precision/
recall/F1 by hand — the numbers here are exactly what Ludwig itself would
report.

Class-index correctness: Ludwig's `confusion_matrix`/`per_class_stats` are
already labeled with `idx2str` names, not raw indices, because they're built
with `labels=train_set_metadata["idx2str"]` inside Ludwig itself. `topErrors`
below compares raw target/prediction *strings* for the same reason — this
sidesteps the whole "which index does Ludwig vs. Postgres think class N is"
class of bug (see `label_classes` in server/db/schema.ts) by never handling a
bare index at all.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# report.json is stored whole in Postgres (server/db/schema.ts's
# runEvaluations.report jsonb column) — cap it.
MAX_TOP_ERRORS = 500
MAX_CLASSES_FOR_MATRIX = 200


def _pick_eval_split(df: pd.DataFrame, split_column: str) -> tuple[pd.DataFrame, str]:
    """test -> validation -> whole dataset. Mirrors the fallback ladder in
    tasks/export.py's `_build_golden_sample`."""
    if split_column in df.columns:
        test_rows = df[df[split_column] == "test"]
        if len(test_rows) > 0:
            return test_rows, "test"
        validation_rows = df[df[split_column] == "validation"]
        if len(validation_rows) > 0:
            return validation_rows, "validation"
    return df, "full"


def _metric(feature_stats: dict[str, Any], key: str) -> float | None:
    value = feature_stats.get(key)
    return round(float(value), 4) if value is not None else None


def _top_errors(
    eval_rows: pd.DataFrame,
    predictions: pd.DataFrame,
    output_column: str,
    feature_name: str,
    item_id_column: str,
) -> list[dict[str, Any]]:
    """Misclassified rows, most-confident-wrong-answer first — a confident
    mistake is the most informative one to show a user — capped at
    MAX_TOP_ERRORS."""
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


def build_evaluation_report(
    model, df: pd.DataFrame, split_column: str, item_id_column: str
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    """Evaluate a trained model against its test split (falling back to
    validation, then the whole dataset — see `_pick_eval_split`) and return
    `(report, predictions_for_download)`.

    `report` is the bounded JSON document persisted by the gateway
    (`runEvaluations.report`); `predictions_for_download` is the full
    per-row prediction set for the accompanying predictions.parquet, which
    the gateway never reads.

    Returns None if there's nothing to evaluate (an empty dataset).
    """
    output_feature = model.config_obj.output_features[0]
    eval_rows, split_used = _pick_eval_split(df, split_column)
    if len(eval_rows) == 0:
        return None

    eval_stats, predictions, _ = model.evaluate(
        dataset=eval_rows, collect_predictions=True, collect_overall_stats=True
    )
    feature_stats = eval_stats.get(output_feature.name, {})

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "split": split_used,
        "outputFeature": output_feature.name,
        "outputType": output_feature.type,
        "rowCount": len(eval_rows),
    }

    if output_feature.type == "category":
        idx2str: list[str] = list((model.training_set_metadata or {}).get(output_feature.name, {}).get("idx2str", []))
        report["idx2str"] = idx2str

        truncated = len(idx2str) > MAX_CLASSES_FOR_MATRIX
        report["truncated"] = truncated

        confusion_matrix = feature_stats.get("confusion_matrix")
        if confusion_matrix is not None and not truncated:
            report["confusionMatrix"] = confusion_matrix

        per_class_stats = feature_stats.get("per_class_stats")
        if per_class_stats is not None and not truncated:
            report["perClass"] = {
                str(label): {
                    "precision": round(float(stats["precision"]), 4),
                    "recall": round(float(stats["recall"]), 4),
                    "f1": round(float(stats["f1_score"]), 4),
                    "support": int(stats["true_positives"] + stats["false_negatives"]),
                }
                for label, stats in per_class_stats.items()
            }

        overall_stats = feature_stats.get("overall_stats", {})
        report["overall"] = {
            "accuracy": _metric(overall_stats, "token_accuracy"),
            "macroF1": _metric(overall_stats, "avg_f1_score_macro"),
        }
        report["topErrors"] = _top_errors(
            eval_rows, predictions, output_feature.column, output_feature.name, item_id_column
        )

    elif output_feature.type == "number":
        report["overall"] = {
            "mae": _metric(feature_stats, "mean_absolute_error"),
            "rmse": _metric(feature_stats, "root_mean_squared_error"),
            "r2": _metric(feature_stats, "r2"),
        }

    else:
        # Sequence/text outputs (token_classification and the LLM-backed
        # experimental tasks) — no dedicated evaluation UI yet. Report
        # whatever scalar metrics Ludwig already computed, nothing more.
        report["overall"] = {
            key: round(float(value), 4) for key, value in feature_stats.items() if isinstance(value, (int, float))
        }

    predictions_for_download = predictions.copy()
    if item_id_column in eval_rows.columns:
        predictions_for_download[item_id_column] = eval_rows[item_id_column].reset_index(drop=True).values
    if output_feature.column in eval_rows.columns:
        predictions_for_download[f"{output_feature.name}_actual"] = (
            eval_rows[output_feature.column].reset_index(drop=True).values
        )

    return report, predictions_for_download
