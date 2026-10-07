"""`YoloLoadedModel`: a trained Ultralytics model behind the framework-neutral `LoadedModel` interface.

The prediction frame uses the same column convention as every backend (`{feature}_predictions`,
`{feature}_probabilities`, plus `{feature}_probability` for `services.evaluate.top_errors`), so batch results,
the evaluation report and the export golden sample all work on it unchanged.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from theseus.backends.base import LoadedModel, OutputSpec
from theseus.backends.yolo import runtime
from theseus.backends.yolo.data import split_s3_uri
from theseus.services import storage
from theseus.services.predict import ClassificationClass, InferenceOutput


class YoloLoadedModel(LoadedModel):
    def __init__(self, model: Any, meta: dict[str, Any]) -> None:
        """`model` is a loaded `ultralytics.YOLO`; `meta` is the parsed `names.json`."""
        self._model = model
        self._meta = meta
        self._image_size: int = meta["imgsz"]
        self._name: str = meta["outputName"]
        # Serving shares one cached instance between requests, and an Ultralytics predictor is not thread-safe.
        self._lock = threading.Lock()
        self.input_columns = [meta["imageColumn"]]
        self.output = OutputSpec(name=self._name, kind="classification", labels=list(meta["names"]))

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def yolo(self) -> Any:
        return self._model

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        paths = [str(p) for p in frame[self.input_columns[0]]]
        labels = self.output.labels or []
        with tempfile.TemporaryDirectory() as scratch:
            sources = [_localize(p, scratch, i) for i, p in enumerate(paths)]
            with self._lock:
                results = self._model.predict(sources, imgsz=self._image_size, device=runtime.device(), verbose=False)

        probabilities = [[float(p) for p in r.probs.data.tolist()] for r in results]
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
        classes: list[ClassificationClass] = [
            {"label": label, "confidence": round(p, 4)} for label, p in self._confidences(predictions).items()
        ]
        classes.sort(key=lambda c: c["confidence"], reverse=True)
        return {"kind": "classification", "feature": self._name, "classes": classes[:top_k]}

    def golden_prediction(self, predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, float]:
        kept = {label: round(p, 4) for label, p in self._confidences(predictions).items() if p >= threshold}
        return dict(sorted(kept.items(), key=lambda item: item[1], reverse=True))

    def close(self) -> None:
        del self._model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: S110 - best effort, only frees cached GPU memory
            pass


def _localize(path: str, scratch: str, index: int) -> str:
    """An `s3://` image URI (a batch frame built from a snapshot) is fetched to a local file; a local path
    (the inference endpoint's upload) is used as it is."""
    if not path.startswith("s3://"):
        return path
    bucket, key = split_s3_uri(path)
    local = os.path.join(scratch, f"{index}{os.path.splitext(key)[1] or '.jpg'}")
    return storage.download_file(bucket, key, local)


def load(model_dir: str) -> YoloLoadedModel:
    runtime.configure()
    from ultralytics import YOLO

    root = Path(model_dir)
    meta = json.loads((root / runtime.METADATA_FILENAME).read_text(encoding="utf-8"))
    return YoloLoadedModel(YOLO(str(root / runtime.WEIGHTS_FILENAME)), meta)
