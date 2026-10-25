"""The preprocessing.json that makes a bare scikit-learn ONNX export usable.

The graph does its own preprocessing (imputation and scaling of a table, tokenising and TF-IDF of a text), so the
manifest asks a client to do nothing to the data: pass raw numbers, or the raw string. That is why a tabular input
carries no `numberNormalization`, which the generated clients would otherwise apply a second time.

The class list MUST come from `names.json` (written at train time from the fitted model's `classes_`), NEVER from the
label_classes table: Postgres has no idea which index a run assigned each class, so guessing wrong silently mislabels
every prediction a generated client makes.
"""

from typing import Any

from omnitrain.plugins.backends.scikit.estimators import METADATA_FILENAME  # noqa: F401  re-exported for the backend


def build_preprocessing_manifest(run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: turn the training-set metadata into the preprocessing manifest."""
    if meta is None:
        raise ValueError(f"Run {run_id} has no {METADATA_FILENAME} to build a preprocessing manifest from")

    if meta["inputKind"] == "text":
        inputs: list[dict[str, Any]] = [
            {
                "name": column,
                "type": "text",
                "column": column,
                # Tokenising, lower-casing and TF-IDF weighting are inside the graph. The Tokenizer operator it uses
                # is onnxruntime's own (domain com.microsoft), so no other ONNX runtime will load the model.
                "textInput": {"encoding": "utf-8", "shape": "[batch, 1] string tensor", "runtime": "onnxruntime"},
            }
            for column in meta["inputColumns"]
        ]
    else:
        # One float input per column; a missing value is NaN, which the graph imputes.
        inputs = [{"name": column, "type": "number", "column": column} for column in meta["inputColumns"]]

    if meta["task"] == "classify":
        output: dict[str, Any] = {
            "name": meta["outputName"],
            "type": "category",
            "column": meta["outputName"],
            "classes": list(meta["names"]),
        }
    else:
        output = {"name": meta["outputName"], "type": "number", "column": meta["outputName"]}
    return {"schemaVersion": 1, "runId": run_id, "inputs": inputs, "outputs": [output]}
