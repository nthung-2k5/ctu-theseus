"""`VideoLoadedModel`: a trained video classifier behind the framework-neutral `LoadedModel` interface.

The prediction frame follows the convention every classifier uses (`{feature}_predictions`, `{feature}_probabilities`,
plus `{feature}_probability` for `services.evaluate.top_errors`), so batch results, the evaluation report and the
export golden sample work on it unchanged. A video is scored as `clipsPerVideo` clips spread across it, with the
softmax probabilities averaged.
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from omnitrain.backends.base import LoadedModel, OutputSpec
from omnitrain.backends.video import frames, network, runtime
from omnitrain.backends.video.data import split_s3_uri
from omnitrain.services import storage
from omnitrain.services.predict import ClassificationClass, InferenceOutput


def clip_spec_from_meta(meta: dict[str, Any]) -> frames.ClipSpec:
    return frames.ClipSpec(
        num_frames=meta["numFrames"],
        stride=meta["frameStride"],
        size=meta["frameSize"],
        mean=tuple(meta["mean"]),
        std=tuple(meta["std"]),
    )


class VideoLoadedModel(LoadedModel):
    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        """`net` is the trained network; `meta` is the parsed `names.json`."""
        self._net = net.eval().to(runtime.device())
        self._meta = meta
        self._spec = clip_spec_from_meta(meta)
        self._name: str = meta["outputName"]
        # Serving shares one cached instance between requests; keep one forward pass at a time on it.
        self._lock = threading.Lock()
        self.input_columns = [meta["clipColumn"]]
        self.output = OutputSpec(name=self._name, kind="classification", labels=list(meta["names"]))

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    def spec(self) -> frames.ClipSpec:
        return self._spec

    def probabilities(self, path: str, clips: int | None = None) -> list[float]:
        """Class probabilities for one local video file, averaged over its clips."""
        count = clips or self._meta["clipsPerVideo"]
        batch = torch.stack([frames.to_tensor(c, self._spec) for c in frames.load_clips(path, self._spec, count)])
        with self._lock, torch.no_grad():
            probs = torch.softmax(self._net(batch.to(runtime.device())), dim=1).mean(dim=0)
        return [float(p) for p in probs.tolist()]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        labels = self.output.labels or []
        paths = [str(p) for p in frame[self.input_columns[0]]]
        probabilities: list[list[float]] = []
        with tempfile.TemporaryDirectory() as scratch:
            for i, path in enumerate(paths):
                probabilities.append(self.probabilities(_localize(path, scratch, i)))
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
        del self._net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _localize(path: str, scratch: str, index: int) -> str:
    """An `s3://` clip URI (a batch frame built from a snapshot) is fetched to a local file; a local path (the
    inference endpoint's upload) is used as it is."""
    if not path.startswith("s3://"):
        return path
    bucket, key = split_s3_uri(path)
    return storage.download_file(bucket, key, os.path.join(scratch, f"{index}{os.path.splitext(key)[1] or '.mp4'}"))


def load(model_dir: str) -> VideoLoadedModel:
    root = Path(model_dir)
    meta = json.loads((root / runtime.METADATA_FILENAME).read_text(encoding="utf-8"))
    net = network.build_model(meta["model"], len(meta["names"]), pretrained=False)
    # weights_only: the file is a plain state dict, and nothing here should ever unpickle arbitrary objects.
    net.load_state_dict(torch.load(root / runtime.WEIGHTS_FILENAME, map_location="cpu", weights_only=True))
    return VideoLoadedModel(net, meta)
