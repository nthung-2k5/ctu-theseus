"""`SklearnLoadedModel`: a trained scikit-learn pipeline behind the framework-neutral `LoadedModel` interface.

The prediction frame uses the convention every backend shares: `{feature}_predictions` (the class name, or the number)
and, for a classifier, `{feature}_probabilities` (one per class, in the model's own `classes_` order) plus
`{feature}_probability` (the winner's, for `domain.evaluate.top_errors`). The class names come from `names.json`,
which `train.py` wrote from the fitted model's `classes_`; the `label_classes` table is never consulted.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from omnitrain.domain.predict import ClassificationClass, InferenceOutput
from omnitrain.plugins.backends.base import LoadedModel, OutputSpec
from omnitrain.plugins.backends.scikit import data, estimators


class SklearnLoadedModel(LoadedModel):
    def __init__(self, estimator: Any, meta: dict[str, Any]) -> None:
        """`estimator` is a fitted pipeline; `meta` is the parsed `names.json`."""
        self._estimator = estimator
        self._meta = meta
        self._kind: str = meta["task"]
        self._name: str = meta["outputName"]
        self.input_columns = list(meta["inputColumns"])
        self.output = OutputSpec(
            name=self._name,
            kind="classification" if self._kind == "classify" else "regression",
            labels=list(meta["names"]) if self._kind == "classify" else None,
        )

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def estimator(self) -> Any:
        return self._estimator

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = data.features(frame, self._meta["inputKind"], self.input_columns)
        if self._kind == "regress":
            return pd.DataFrame({f"{self._name}_predictions": [float(v) for v in self._estimator.predict(x)]})
        labels = self.output.labels or []
        probabilities = [[float(p) for p in row] for row in self._estimator.predict_proba(x)]
        top = [max(range(len(p)), key=p.__getitem__) for p in probabilities]
        return pd.DataFrame(
            {
                f"{self._name}_predictions": [labels[i] for i in top],
                f"{self._name}_probabilities": probabilities,
                f"{self._name}_probability": [p[i] for p, i in zip(probabilities, top, strict=True)],
            }
        )

    def _confidences(self, predictions: pd.DataFrame) -> dict[str, float]:
        probs = predictions[f"{self._name}_probabilities"].iloc[0]
        labels = self.output.labels or []
        return {labels[i]: float(p) for i, p in enumerate(probs) if i < len(labels)}

    def to_output(
        self, predictions: pd.DataFrame, *, top_k: int = 100, input_tokens: list[str] | None = None
    ) -> InferenceOutput:
        if self._kind == "regress":
            value = float(predictions[f"{self._name}_predictions"].iloc[0])
            return {"kind": "regression", "feature": self._name, "value": round(value, 4)}
        classes: list[ClassificationClass] = [
            {"label": label, "confidence": round(p, 4)} for label, p in self._confidences(predictions).items()
        ]
        classes.sort(key=lambda c: c["confidence"], reverse=True)
        return {"kind": "classification", "feature": self._name, "classes": classes[:top_k]}

    def golden_prediction(self, predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, float]:
        if self._kind == "regress":
            return {self._name: round(float(predictions[f"{self._name}_predictions"].iloc[0]), 4)}
        kept = {label: round(p, 4) for label, p in self._confidences(predictions).items() if p >= threshold}
        return dict(sorted(kept.items(), key=lambda item: item[1], reverse=True))


def load(model_dir: str) -> SklearnLoadedModel:
    meta = json.loads((Path(model_dir) / estimators.METADATA_FILENAME).read_text(encoding="utf-8"))
    return SklearnLoadedModel(estimators.load(model_dir), meta)
