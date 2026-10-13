"""`YoloLoadedModel`: a trained Ultralytics model behind the framework-neutral `LoadedModel` interface.

Classification and detection share this class; `meta["task"]` (written at train time, see `train.py`) says which.

Classification's prediction frame uses the same column convention as every backend (`{feature}_predictions`,
`{feature}_probabilities`, plus `{feature}_probability` for `domain.evaluate.top_errors`), so batch results,
the evaluation report and the export golden sample all work on it unchanged.

Detection has one row per image holding a LIST of detections, in the convention bounding-box annotations use
(domain/boxes.py): the top-left corner and the size, as fractions of the image. `{feature}_predictions` is that
list (most confident first) and `{feature}_image_size` is `[width, height]` in pixels.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from omnitrain.backends.base import LoadedModel, OutputSpec
from omnitrain.backends.yolo import runtime
from omnitrain.backends.yolo.data import split_s3_uri
from omnitrain.domain.predict import ClassificationClass, Detection, InferenceOutput
from omnitrain.services import storage

# Detections below this confidence are not returned when serving. Evaluation uses a far lower one (mAP needs the
# whole precision/recall curve); see `evaluate.py`.
DETECT_CONFIDENCE = 0.25


class YoloLoadedModel(LoadedModel):
    def __init__(self, model: Any, meta: dict[str, Any]) -> None:
        """`model` is a loaded `ultralytics.YOLO`; `meta` is the parsed `names.json`."""
        self._model = model
        self._meta = meta
        self._kind: str = meta["task"]
        self._image_size: int = meta["imgsz"]
        self._name: str = meta["outputName"]
        # Serving shares one cached instance between requests, and an Ultralytics predictor is not thread-safe.
        self._lock = threading.Lock()
        self.input_columns = [meta["imageColumn"]]
        self.output = OutputSpec(
            name=self._name,
            kind="detections" if self._kind == "detect" else "classification",
            labels=list(meta["names"]),
        )

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def yolo(self) -> Any:
        return self._model

    @property
    def lock(self) -> threading.Lock:
        """Held around anything that drives the underlying `YOLO`, which is not safe to share between threads."""
        return self._lock

    def predict(self, frame: pd.DataFrame, *, confidence: float = DETECT_CONFIDENCE) -> pd.DataFrame:
        """`confidence` is the detection threshold; classification has no such thing."""
        paths = [str(p) for p in frame[self.input_columns[0]]]
        options: dict[str, Any] = {"imgsz": self._image_size, "device": runtime.device(), "verbose": False}
        if self._kind == "detect":
            options["conf"] = confidence
        with tempfile.TemporaryDirectory() as scratch:
            sources = [_localize(p, scratch, i) for i, p in enumerate(paths)]
            with self._lock:
                results = self._model.predict(sources, **options)
        if self._kind == "detect":
            return self._detections_frame(results)
        return self._classification_frame(results)

    def _classification_frame(self, results: Any) -> pd.DataFrame:
        labels = self.output.labels or []
        probabilities = [[float(p) for p in r.probs.data.tolist()] for r in results]
        top = [max(range(len(p)), key=p.__getitem__) for p in probabilities]
        return pd.DataFrame(
            {
                f"{self._name}_predictions": [labels[i] for i in top],
                f"{self._name}_probabilities": probabilities,
                f"{self._name}_probability": [p[i] for p, i in zip(probabilities, top, strict=True)],
            }
        )

    def _detections_frame(self, results: Any) -> pd.DataFrame:
        labels = self.output.labels or []
        rows: list[list[Detection]] = []
        sizes: list[list[int]] = []
        for r in results:
            height, width = r.orig_shape
            detections: list[Detection] = []
            for (x1, y1, x2, y2), confidence, class_index in zip(
                r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True
            ):
                x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
                detections.append(
                    {
                        "label": labels[int(class_index)],
                        "confidence": float(confidence),
                        "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                    }
                )
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            rows.append(detections)
            sizes.append([int(width), int(height)])
        return pd.DataFrame({f"{self._name}_predictions": rows, f"{self._name}_image_size": sizes})

    def _confidences(self, predictions: pd.DataFrame) -> dict[str, float]:
        probs = predictions[f"{self._name}_probabilities"].iloc[0]
        labels = self.output.labels or []
        return {labels[i]: float(p) for i, p in enumerate(probs) if i < len(labels)}

    def to_output(
        self, predictions: pd.DataFrame, *, top_k: int = 100, input_tokens: list[str] | None = None
    ) -> InferenceOutput:
        if self._kind == "detect":
            width, height = predictions[f"{self._name}_image_size"].iloc[0]
            detections: list[Detection] = [
                {
                    "label": d["label"],
                    "confidence": round(d["confidence"], 4),
                    "box": {
                        "x": round(d["box"]["x"], 6),
                        "y": round(d["box"]["y"], 6),
                        "w": round(d["box"]["w"], 6),
                        "h": round(d["box"]["h"], 6),
                    },
                }
                for d in predictions[f"{self._name}_predictions"].iloc[0][:top_k]
            ]
            return {
                "kind": "detections",
                "feature": self._name,
                "imageWidth": int(width),
                "imageHeight": int(height),
                "detections": detections,
            }
        classes: list[ClassificationClass] = [
            {"label": label, "confidence": round(p, 4)} for label, p in self._confidences(predictions).items()
        ]
        classes.sort(key=lambda c: c["confidence"], reverse=True)
        return {"kind": "classification", "feature": self._name, "classes": classes[:top_k]}

    def golden_prediction(self, predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, float]:
        if self._kind == "detect":
            # The flat `{label: value}` shape every consumer expects, so: the best confidence seen per class.
            best: dict[str, float] = {}
            for d in predictions[f"{self._name}_predictions"].iloc[0]:
                best[d["label"]] = max(best.get(d["label"], 0.0), round(d["confidence"], 4))
            return dict(sorted(((k, v) for k, v in best.items() if v >= threshold), key=lambda kv: kv[1], reverse=True))
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
