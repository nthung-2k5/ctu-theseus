"""`PytorchLoadedModel`: a trained PyTorch model behind the framework-neutral `LoadedModel` interface.

The prediction frame follows the convention every backend shares: `{feature}_predictions` (the class name, or the
number) and, for a classifier, `{feature}_probabilities` (one per class, in the model's own order) plus
`{feature}_probability` (the winner's, for `domain.evaluate.top_errors`), so batch results, the evaluation report and
the export golden sample work on it unchanged.

What differs between the kinds of model is only how a frame becomes network output (`outputs`), which reads the
spec stored in `names.json`; the rest is shared. Images go through `images.preprocess`, a sound through `audio`, a text
through `text.encode`: the same steps an export client applies.
"""

import json
import os
import tempfile
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from omnitrain.domain.predict import ClassificationClass, InferenceOutput
from omnitrain.plugins.backends.base import LoadedModel, OutputSpec
from omnitrain.plugins.backends.pytorch import audio, images, network, runtime, tabular, text
from omnitrain.plugins.backends.pytorch.data import split_s3_uri
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, ImageSpec, TabularSpec, TextSpec
from omnitrain.services import storage

# Rows scored per forward pass, which bounds memory however many rows a frame has.
IMAGE_PREDICT_BATCH = 32
TABULAR_PREDICT_BATCH = 4096
AUDIO_PREDICT_BATCH = 8
TEXT_PREDICT_BATCH = 256
# What an evaluation feeds `predict` at a time: the model's own forward batch.
PREDICT_BATCH = IMAGE_PREDICT_BATCH


class PytorchLoadedModel(LoadedModel):
    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        """`net` is the trained network; `meta` is the parsed `names.json`."""
        self._net = net.eval().to(runtime.device())
        self._meta = meta
        self._kind: str = meta["task"]
        self._name: str = meta["outputName"]
        # Serving shares one cached instance between requests; keep one forward pass at a time on it.
        self._lock = threading.Lock()
        self.output = OutputSpec(
            name=self._name,
            kind="classification" if self._kind == "classify" else "regression",
            labels=list(meta["names"]) if self._kind == "classify" else None,
        )

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    def predict_batch(self) -> int:
        return PREDICT_BATCH

    def _forward(self, batch: torch.Tensor) -> np.ndarray:
        with self._lock, torch.no_grad():
            return self._net(batch.to(runtime.device())).float().cpu().numpy()

    @abstractmethod
    def outputs(self, frame: pd.DataFrame) -> np.ndarray:
        """The network's raw output for each row of `frame`: class logits `[rows, classes]`, or one number per row."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self.outputs(frame)
        if self._kind == "regress":
            return pd.DataFrame({f"{self._name}_predictions": [float(v) for v in out.reshape(-1)]})
        labels = self.output.labels or []
        probabilities = torch.softmax(torch.from_numpy(out), dim=1).tolist()
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

    def close(self) -> None:
        del self._net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageModel(PytorchLoadedModel):
    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        super().__init__(net, meta)
        self._spec = ImageSpec.from_meta(meta)
        self.input_columns = [meta["imageColumn"]]

    @property
    def spec(self) -> ImageSpec:
        return self._spec

    def outputs(self, frame: pd.DataFrame) -> np.ndarray:
        paths = [str(p) for p in frame[self.input_columns[0]]]
        out: list[np.ndarray] = []
        with tempfile.TemporaryDirectory() as scratch:
            local = [_localize(p, scratch, i) for i, p in enumerate(paths)]
            for start in range(0, len(local), IMAGE_PREDICT_BATCH):
                batch = torch.stack(
                    [
                        torch.from_numpy(images.preprocess(p, self._spec))
                        for p in local[start : start + IMAGE_PREDICT_BATCH]
                    ]
                )
                out.append(self._forward(batch))
        return np.concatenate(out) if out else np.zeros((0, len(self.output.labels or [])), dtype=np.float32)


class TabularModel(PytorchLoadedModel):
    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        super().__init__(net, meta)
        self._spec = TabularSpec.from_meta(meta)
        self.input_columns = list(self._spec.columns)

    @property
    def spec(self) -> TabularSpec:
        return self._spec

    @property
    def predict_batch(self) -> int:
        return TABULAR_PREDICT_BATCH

    def outputs(self, frame: pd.DataFrame) -> np.ndarray:
        x = tabular.features(frame, self._spec.columns)
        out = [
            self._forward(torch.from_numpy(x[start : start + TABULAR_PREDICT_BATCH]))
            for start in range(0, len(x), TABULAR_PREDICT_BATCH)
        ]
        if out:
            return np.concatenate(out)
        return np.zeros((0,) if self._kind == "regress" else (0, len(self.output.labels or [])), dtype=np.float32)


class AudioModel(PytorchLoadedModel):
    """Scores a recording as `clipsPerAudio` windows spread over it, averaging their class probabilities (as the video
    backend does for clips). `outputs` returns the log of the averaged probabilities, which `predict` turns back into
    exactly those probabilities with its softmax."""

    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        super().__init__(net, meta)
        self._spec = AudioSpec.from_meta(meta)
        self.input_columns = [meta["audioColumn"]]

    @property
    def spec(self) -> AudioSpec:
        return self._spec

    @property
    def predict_batch(self) -> int:
        return AUDIO_PREDICT_BATCH

    def outputs(self, frame: pd.DataFrame) -> np.ndarray:
        spec = self._spec
        rows: list[np.ndarray] = []
        with tempfile.TemporaryDirectory() as scratch:
            for i, path in enumerate(str(p) for p in frame[self.input_columns[0]]):
                wave = audio.load_waveform(_localize(path, scratch, i, ".wav"), spec)
                starts = audio.window_starts(len(wave), spec.length, spec.clips)
                windows = np.stack([audio.spectrogram(audio.cut(wave, s, spec.length), spec) for s in starts])
                logits = torch.from_numpy(self._forward(torch.from_numpy(windows[:, None])))
                mean = torch.softmax(logits, dim=1).mean(dim=0)
                rows.append(torch.log(mean.clamp_min(1e-12)).numpy())
        return np.stack(rows) if rows else np.zeros((0, len(self.output.labels or [])), dtype=np.float32)


class TextModel(PytorchLoadedModel):
    def __init__(self, net: torch.nn.Module, meta: dict[str, Any]) -> None:
        super().__init__(net, meta)
        self._spec = TextSpec.from_meta(meta)
        self.input_columns = [meta["textColumn"]]

    @property
    def spec(self) -> TextSpec:
        return self._spec

    @property
    def predict_batch(self) -> int:
        return TEXT_PREDICT_BATCH

    def outputs(self, frame: pd.DataFrame) -> np.ndarray:
        ids = text.encode(text.clean(frame[self.input_columns[0]]), self._spec)
        out = [
            self._forward(torch.from_numpy(ids[start : start + TEXT_PREDICT_BATCH]))
            for start in range(0, len(ids), TEXT_PREDICT_BATCH)
        ]
        return np.concatenate(out) if out else np.zeros((0, len(self.output.labels or [])), dtype=np.float32)


def _localize(path: str, scratch: str, index: int, default_ext: str = ".jpg") -> str:
    """An `s3://` URI (a batch frame built from a snapshot) is fetched to a local file; a local path (the inference
    endpoint's upload) is used as it is."""
    if not path.startswith("s3://"):
        return path
    bucket, key = split_s3_uri(path)
    return storage.download_file(
        bucket, key, os.path.join(scratch, f"{index}{os.path.splitext(key)[1] or default_ext}")
    )


def build_network(meta: dict[str, Any]) -> torch.nn.Module:
    """The untrained network a `names.json` describes, ready to have its weights loaded."""
    if meta["inputKind"] == "tabular":
        regress = meta["task"] == "regress"
        return network.build_tabular(
            TabularSpec.from_meta(meta),
            list(meta["network"]["hidden"]),
            float(meta["network"]["dropout"]),
            len(meta["names"]),
            regress=regress,
        )
    if meta["inputKind"] == "audio":
        return network.build_audio(meta["model"], len(meta["names"]), pretrained=False)
    if meta["inputKind"] == "text":
        spec = TextSpec.from_meta(meta)
        return network.build_text(meta["model"], spec.size, len(meta["names"]), float(meta["network"]["dropout"]))
    return network.build_model(meta["model"], len(meta["names"]), pretrained=False)


def wrap(net: torch.nn.Module, meta: dict[str, Any]) -> PytorchLoadedModel:
    """`net` as the loaded model its `names.json` calls for."""
    kinds = {"tabular": TabularModel, "audio": AudioModel, "image": ImageModel, "text": TextModel}
    return kinds[meta["inputKind"]](net, meta)


def load(model_dir: str) -> PytorchLoadedModel:
    root = Path(model_dir)
    meta = json.loads((root / runtime.METADATA_FILENAME).read_text(encoding="utf-8"))
    net = build_network(meta)
    # weights_only: the file is a plain state dict, and nothing here should ever unpickle arbitrary objects.
    net.load_state_dict(torch.load(root / runtime.WEIGHTS_FILENAME, map_location="cpu", weights_only=True))
    return wrap(net, meta)
