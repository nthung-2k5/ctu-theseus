"""The evaluation report for a YOLO classifier, in the same shape Ludwig's is (the web panel and
`run_evaluations` read it without knowing which backend wrote it).

Ultralytics reports no per-class statistics through its public API, so they are computed here from the model's
own predictions on the evaluation split. Every label in the report is a class NAME taken from the model
(`names.json`), never an index, and never from the `label_classes` table (see services/evaluate.py).
"""

from typing import Any

import numpy as np
import pandas as pd

from theseus.backends.base import EvalResult
from theseus.backends.yolo.model import YoloLoadedModel
from theseus.services.evaluate import MAX_CLASSES_FOR_MATRIX, pick_eval_split, top_errors

# Images predicted per call, so a large split is neither one giant batch nor one call per image.
PREDICT_BATCH = 32


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(model: YoloLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    labels = list(model.output.labels or [])
    image_column = model.input_columns[0]
    name = model.output.name
    label_column = model.meta["outputName"]

    eval_rows, split_used = pick_eval_split(df, split_column)
    # A row with no image, or a class this model has never heard of, has nothing to be scored against.
    eval_rows = eval_rows[eval_rows[image_column].notna() & eval_rows[label_column].isin(labels)].reset_index(drop=True)
    if len(eval_rows) == 0:
        return None

    batches = [
        model.predict(eval_rows.iloc[start : start + PREDICT_BATCH][[image_column]])
        for start in range(0, len(eval_rows), PREDICT_BATCH)
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

    truncated = len(labels) > MAX_CLASSES_FOR_MATRIX
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
