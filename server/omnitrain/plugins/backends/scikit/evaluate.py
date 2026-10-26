"""The evaluation report for a scikit-learn model.

Classification: the shared report every one-class-per-input backend uses (`domain/classification_report.py`).
Regression: the shared number report (`domain/regression_report.py`), `overall` holding MAE, RMSE and R².
"""

import pandas as pd

from omnitrain.domain.classification_report import evaluate_classification
from omnitrain.domain.evaluate import MAX_CLASSES_FOR_MATRIX
from omnitrain.domain.regression_report import evaluate_regression
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
    return evaluate_regression(
        model, df, split_column, item_id_column, label_column=label_column, batch_size=PREDICT_BATCH
    )
