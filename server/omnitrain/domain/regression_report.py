"""The regression evaluation report, shared by every backend whose model predicts one number per row.

The report has a fixed shape for a number output (the web panel and `run_evaluations` read it without knowing which
backend wrote it): `overall` holds MAE, RMSE and R².

`model` is any `LoadedModel` whose input columns hold what `predict` reads and whose prediction frame has a
`{feature}_predictions` column of numbers.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from omnitrain.domain.evaluate import pick_eval_split
from omnitrain.plugins.backends.base import EvalResult, LoadedModel


def _round(value: float) -> float | None:
    return round(float(value), 4) if math.isfinite(value) else None


def evaluate_regression(
    model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str, *, label_column: str, batch_size: int
) -> EvalResult | None:
    name = model.output.name
    eval_rows, split_used = pick_eval_split(df, split_column)
    actual_all = pd.to_numeric(eval_rows[label_column], errors="coerce")
    eval_rows = eval_rows[actual_all.notna()].reset_index(drop=True)
    if len(eval_rows) == 0:
        return None
    actual = pd.to_numeric(eval_rows[label_column]).to_numpy(dtype=float)

    batches = [
        model.predict(eval_rows.iloc[start : start + batch_size][model.input_columns])
        for start in range(0, len(eval_rows), batch_size)
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
