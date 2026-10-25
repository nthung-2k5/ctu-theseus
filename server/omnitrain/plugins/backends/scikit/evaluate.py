"""The evaluation report for a scikit-learn model.

Classification: the shared report every one-class-per-input backend uses (`domain/classification_report.py`).
Regression: Ludwig's shape for a number output, `overall` holding MAE, RMSE and R².
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from omnitrain.domain.classification_report import evaluate_classification
from omnitrain.domain.evaluate import MAX_CLASSES_FOR_MATRIX, pick_eval_split
from omnitrain.plugins.backends.base import EvalResult
from omnitrain.plugins.backends.scikit.model import SklearnLoadedModel

# Rows predicted per call, so a large split is neither one giant batch nor one call per row.
PREDICT_BATCH = 2048


def evaluate(model: SklearnLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    label_column = model.meta["outputName"]
    if model.meta["task"] == "classify":
        return evaluate_classification(
            model, df, split_column, item_id_column,
            label_column=label_column, batch_size=PREDICT_BATCH, max_classes_for_matrix=MAX_CLASSES_FOR_MATRIX,
            # A table row with an empty cell is still scored: the model imputes it, or handles it natively.
            allow_missing_inputs=model.meta["inputKind"] == "record",
        )  # fmt: skip
    return _evaluate_regression(model, df, split_column, item_id_column, label_column)


def _round(value: float) -> float | None:
    return round(float(value), 4) if math.isfinite(value) else None


def _evaluate_regression(
    model: SklearnLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str, label_column: str
) -> EvalResult | None:
    name = model.output.name
    eval_rows, split_used = pick_eval_split(df, split_column)
    actual_all = pd.to_numeric(eval_rows[label_column], errors="coerce")
    eval_rows = eval_rows[actual_all.notna()].reset_index(drop=True)
    if len(eval_rows) == 0:
        return None
    actual = pd.to_numeric(eval_rows[label_column]).to_numpy(dtype=float)

    batches = [
        model.predict(eval_rows.iloc[start : start + PREDICT_BATCH][model.input_columns])
        for start in range(0, len(eval_rows), PREDICT_BATCH)
    ]
    predictions = pd.concat(batches, ignore_index=True)
    predicted = predictions[f"{name}_predictions"].to_numpy(dtype=float)

    error = predicted - actual
    spread = float(((actual - actual.mean()) ** 2).sum())
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "split": split_used,
        "outputFeature": name,
        "outputType": "number",
        "rowCount": len(eval_rows),
        "overall": {
            "mae": _round(np.abs(error).mean()),
            "rmse": _round(math.sqrt(float((error**2).mean()))),
            # R² is undefined when every actual value is the same.
            "r2": _round(1.0 - float((error**2).sum()) / spread) if spread > 0 else None,
        },
    }

    download = predictions.copy()
    if item_id_column in eval_rows.columns:
        download[item_id_column] = eval_rows[item_id_column].values
    download[f"{name}_actual"] = actual
    return EvalResult(report=report, predictions=download)
