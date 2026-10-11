"""The evaluation report for a YOLO model.

Classification: in the same shape Ludwig's is (the web panel and `run_evaluations` read it without knowing which
backend wrote it).

Detection: `kind: "detection"`, scored by Ultralytics' own validator (COCO-style mAP, which needs the whole
precision/recall curve, so it runs at a very low confidence threshold), plus per-class numbers.

For classification, Ultralytics reports no per-class statistics through its public API, so they are computed
here from the model's own predictions on the evaluation split. Every label in either report is a class NAME taken
from the model (`names.json`), never an index, and never from the `label_classes` table (see services/evaluate.py).
"""

import json
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from omnitrain.backends.base import EvalResult
from omnitrain.backends.yolo import data, runtime
from omnitrain.backends.yolo.model import YoloLoadedModel
from omnitrain.services.boxes import parse_boxes
from omnitrain.services.evaluate import MAX_CLASSES_FOR_MATRIX, pick_eval_split, top_errors

# Images predicted per call, so a large split is neither one giant batch nor one call per image.
PREDICT_BATCH = 32


# mAP integrates precision over recall, so every detection down to this confidence is scored.
EVALUATION_CONFIDENCE = 0.001


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(model: YoloLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    if model.meta["task"] == "detect":
        return _evaluate_detection(model, df, split_column, item_id_column)
    return _evaluate_classification(model, df, split_column, item_id_column)


def _evaluate_classification(
    model: YoloLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
) -> EvalResult | None:
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


def _evaluate_detection(
    model: YoloLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
) -> EvalResult | None:
    labels = list(model.output.labels or [])
    image_column = model.input_columns[0]
    boxes_column = model.meta["outputName"]
    name = model.output.name

    eval_rows, split_used = pick_eval_split(df, split_column)
    # An image nobody labelled (no `boxes` value) has nothing to be scored against; an image labelled as empty does.
    labelled = eval_rows[image_column].notna() & eval_rows[boxes_column].map(
        lambda cell: parse_boxes(cell) is not None
    ).astype(bool)
    eval_rows = eval_rows[labelled].reset_index(drop=True)
    if len(eval_rows) == 0:
        return None

    with tempfile.TemporaryDirectory() as work:
        dataset = data.materialize_detection(
            eval_rows,
            f"{work}/data",
            image_column=image_column,
            boxes_column=boxes_column,
            split_column=split_column,
            item_id_column=item_id_column,
            class_names=labels,
            evaluation_only=True,
        )
        with model.lock:
            metrics = model.yolo.val(
                data=dataset.data_yaml,
                split="val",
                imgsz=model.meta["imgsz"],
                batch=16,
                conf=EVALUATION_CONFIDENCE,
                device=runtime.device(),
                workers=runtime.dataloader_workers(),
                project=f"{work}/runs",
                name="val",
                exist_ok=True,
                plots=False,
                save_json=False,
                verbose=False,
            )

    box = metrics.box
    # The validator only lists classes that occur in the images (`ap_class_index`); `nt_per_class` covers all of them.
    present = {int(c): i for i, c in enumerate(box.ap_class_index)}
    instances = [int(n) for n in metrics.nt_per_class]
    per_class: dict[str, dict[str, Any]] = {}
    for class_index, label in enumerate(labels):
        row = present.get(class_index)
        per_class[label] = {
            "precision": round(float(box.p[row]), 4) if row is not None else 0.0,
            "recall": round(float(box.r[row]), 4) if row is not None else 0.0,
            "ap50": round(float(box.ap50[row]), 4) if row is not None else 0.0,
            "ap5095": round(float(box.ap[row]), 4) if row is not None else 0.0,
            "instances": instances[class_index] if class_index < len(instances) else 0,
        }

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "detection",
        "split": split_used,
        "outputFeature": name,
        "outputType": "detections",
        "rowCount": len(eval_rows),
        "imageCount": len(eval_rows),
        "instanceCount": sum(instances),
        "idx2str": labels,
        "truncated": len(labels) > MAX_CLASSES_FOR_MATRIX,
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map5095": round(float(box.map), 4),
        },
    }
    if not report["truncated"]:
        report["perClass"] = per_class

    # What the model says about each evaluated image at the serving threshold, next to what was labelled.
    batches = [
        model.predict(eval_rows.iloc[start : start + PREDICT_BATCH][[image_column]])
        for start in range(0, len(eval_rows), PREDICT_BATCH)
    ]
    predictions = pd.concat(batches, ignore_index=True)
    download = pd.DataFrame(
        {
            item_id_column: eval_rows[item_id_column].values,
            f"{name}_actual": eval_rows[boxes_column].values,
            f"{name}_predictions": [json.dumps(p, separators=(",", ":")) for p in predictions[f"{name}_predictions"]],
        }
    )
    return EvalResult(report=report, predictions=download)
