"""The evaluation report for a PyTorch model: the shared classification report (confusion matrix, per-class
precision/recall/F1, top errors) or the shared number report (MAE, RMSE, R²)."""

import pandas as pd

from omnitrain.domain.classification_report import evaluate_classification
from omnitrain.domain.evaluate import MAX_CLASSES_FOR_MATRIX
from omnitrain.domain.regression_report import evaluate_regression
from omnitrain.domain.token_report import evaluate_tokens
from omnitrain.plugins.backends.base import EvalResult
from omnitrain.plugins.backends.pytorch.model import PytorchLoadedModel


def evaluate(model: PytorchLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    label_column = model.meta["outputName"]
    if model.meta["task"] == "tag":
        return evaluate_tokens(
            model, df, split_column, item_id_column,
            label_column=label_column, batch_size=model.predict_batch, max_classes_for_matrix=MAX_CLASSES_FOR_MATRIX,
        )  # fmt: skip
    if model.meta["task"] == "regress":
        return evaluate_regression(
            model, df, split_column, item_id_column, label_column=label_column, batch_size=model.predict_batch
        )
    return evaluate_classification(
        model, df, split_column, item_id_column,
        label_column=label_column, batch_size=model.predict_batch, max_classes_for_matrix=MAX_CLASSES_FOR_MATRIX,
        # A table row with an empty cell is still scored: the network fills it in, as it did in training.
        allow_missing_inputs=model.meta["inputKind"] == "tabular",
    )  # fmt: skip
