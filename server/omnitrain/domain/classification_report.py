"""The classification evaluation report, shared by every backend whose model predicts one class per input.

The report has Ludwig's shape (the web panel and `run_evaluations` read it without knowing which backend wrote it).
Ultralytics and the video backend report no per-class statistics through their own APIs, so they are computed here
from the model's predictions on the evaluation split. Every label is a class NAME taken from the model's own
`OutputSpec.labels`, never an index and never from the `label_classes` table (see domain/evaluate.py).

`model` is any `LoadedModel` whose input columns hold what `predict` reads and whose prediction frame follows
the `{feature}_predictions` / `{feature}_probability` convention.
"""

from typing import Any

import numpy as np
import pandas as pd

from omnitrain.domain.evaluate import pick_eval_split, top_errors
from omnitrain.plugins.backends.base import EvalResult, LoadedModel


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_classification(
    model: LoadedModel,
    df: pd.DataFrame,
    split_column: str,
    item_id_column: str,
    *,
    label_column: str,
    batch_size: int,
    max_classes_for_matrix: int,
    allow_missing_inputs: bool = False,
) -> EvalResult | None:
    """`allow_missing_inputs`: score a row even when some input cell is empty, for a model (tabular) that has its own
    way of coping with a missing value. Otherwise such a row has nothing to be predicted from and is left out."""
    labels = list(model.output.labels or [])
    inputs = list(model.input_columns)
    name = model.output.name

    eval_rows, split_used = pick_eval_split(df, split_column)
    # A row with no input, or a class this model has never heard of, has nothing to be scored against.
    keep = eval_rows[label_column].isin(labels)
    if not allow_missing_inputs:
        keep &= eval_rows[inputs].notna().all(axis=1)
    eval_rows = eval_rows[keep].reset_index(drop=True)
    if len(eval_rows) == 0:
        return None

    batches = [
        model.predict(eval_rows.iloc[start : start + batch_size][inputs])
        for start in range(0, len(eval_rows), batch_size)
    ]
    predictions = pd.concat(batches, ignore_index=True)

    index = {label: i for i, label in enumerate(labels)}
    actual = eval_rows[label_column].map(index).to_numpy()
    predicted = predictions[f"{name}_predictions"].map(index).to_numpy()
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)

    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []
    for i, label in enumerate(labels):
        precision = _ratio(true_positive[i], predicted_count[i])
        recall = _ratio(true_positive[i], support[i])
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support[i]),
        }
        # A class that is neither in the split nor predicted says nothing about the model, so it is not averaged in.
        if support[i] or predicted_count[i]:
            f1_scores.append(f1)

    truncated = len(labels) > max_classes_for_matrix
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "split": split_used,
        "outputFeature": name,
        "outputType": "category",
        "rowCount": len(eval_rows),
        "idx2str": labels,
        "truncated": truncated,
        "overall": {
            "accuracy": round(_ratio(true_positive.sum(), matrix.sum()), 4),
            "macroF1": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
        },
        "topErrors": top_errors(eval_rows, predictions, label_column, name, item_id_column),
    }
    if not truncated:
        report["confusionMatrix"] = matrix.tolist()
        report["perClass"] = per_class

    download = predictions.copy()
    if item_id_column in eval_rows.columns:
        download[item_id_column] = eval_rows[item_id_column].values
    download[f"{name}_actual"] = eval_rows[label_column].values
    return EvalResult(report=report, predictions=download)
