"""Ludwig's evaluation report: built on Ludwig's own `ConfusionMatrix`
(`ludwig.utils.eval_utils`, the same class `category_feature.py`'s `calculate_overall_stats` uses
internally) rather than recomputing precision/recall/F1 by hand — the numbers here are exactly
what Ludwig itself would report.

Class-index correctness: Ludwig's `confusion_matrix`/`per_class_stats` are already labeled with
`idx2str` names, not raw indices, because they're built with `labels=train_set_metadata["idx2str"]`
inside Ludwig itself. `top_errors` compares raw target/prediction *strings* for the same reason.
"""

from typing import Any

import pandas as pd

from theseus.backends.base import EvalResult
from theseus.services.evaluate import MAX_CLASSES_FOR_MATRIX, metric, pick_eval_split, top_errors


def evaluate(model: Any, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    """`model` is a loaded `ludwig.api.LudwigModel` (typed `Any`: this module has no ludwig import
    of its own, and the tests exercise it against a lightweight fake with the same surface)."""
    output_feature = model.config_obj.output_features[0]
    eval_rows, split_used = pick_eval_split(df, split_column)
    if len(eval_rows) == 0:
        return None

    eval_stats, predictions, _ = model.evaluate(dataset=eval_rows, collect_predictions=True, collect_overall_stats=True)
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
            "accuracy": metric(overall_stats, "token_accuracy"),
            "macroF1": metric(overall_stats, "avg_f1_score_macro"),
        }
        report["topErrors"] = top_errors(
            eval_rows, predictions, output_feature.column, output_feature.name, item_id_column
        )

    elif output_feature.type == "number":
        report["overall"] = {
            "mae": metric(feature_stats, "mean_absolute_error"),
            "rmse": metric(feature_stats, "root_mean_squared_error"),
            "r2": metric(feature_stats, "r2"),
        }

    else:
        # Sequence/text outputs (token_classification and the LLM-backed experimental tasks) — no
        # dedicated evaluation UI yet. Report whatever scalar metrics Ludwig already computed.
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

    return EvalResult(report=report, predictions=predictions_for_download)
