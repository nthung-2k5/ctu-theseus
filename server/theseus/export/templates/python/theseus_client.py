"""Theseus-generated inference client.

Wraps the exported ONNX model with the preprocessing/postprocessing Ludwig
applied at training time, read from preprocessing.json — Ludwig's exported
ONNX graph is the bare model only (no resize/normalize, no tokenization, no
class-index decoding), so this client reconstructs that step from the
manifest rather than assuming a fixed tensor layout: input/output tensor
names are matched against the graph's actual names at runtime, not
hardcoded, since ONNX export doesn't guarantee a stable naming convention
across Ludwig versions.

Implements preprocessing for:
  - a single `image` input with a `category`/`number` output (image
    classification/regression) — pass a file path to `predict()`.
  - one or more `number` inputs (tabular classification/regression) — pass
    a `{column: value}` dict to `predict()`, one entry per input feature.

Every other input type (text, audio) raises NotImplementedError with a
pointer to preprocessing.json so you can adapt preprocessing yourself —
text tokenization in particular depends on the chosen encoder's own
vocabulary (Ludwig's built-in tokenizer, or a HuggingFace subword
tokenizer for bert/distilbert/roberta), which can't be generically
reconstructed from preprocessing.json alone. See README.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

_HERE = Path(__file__).parent


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class TheseusClient:
    def __init__(
        self,
        model_path: str | Path = _HERE / "model.onnx",
        preprocessing_path: str | Path = _HERE / "preprocessing.json",
    ):
        self.session = ort.InferenceSession(str(model_path))
        with open(preprocessing_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.input_specs: list[dict] = self.manifest["inputs"]
        self.input_spec = self.input_specs[0]
        self.output_spec = self.manifest["outputs"][0]

        onnx_input_names = [i.name for i in self.session.get_inputs()]
        self._onnx_input_names = {
            spec["name"]: self._match_tensor_name(onnx_input_names, spec["column"]) for spec in self.input_specs
        }
        self._onnx_output_name = self._match_tensor_name(
            [o.name for o in self.session.get_outputs()], self.output_spec["column"]
        )
        # Multiple inputs, or a single tabular (number) one, means predict()
        # expects a {column: value} dict rather than a single file path.
        self._is_tabular = len(self.input_specs) > 1 or self.input_spec["type"] == "number"

    @staticmethod
    def _match_tensor_name(candidates: list[str], preferred: str) -> str:
        if len(candidates) == 1:
            return candidates[0]
        for name in candidates:
            if name == preferred or name.startswith((f"{preferred}::", f"{preferred}_")):
                return name
        raise ValueError(
            f"Could not match an ONNX tensor to feature '{preferred}' among {candidates}. "
            "Inspect preprocessing.json and pass the tensor name explicitly if this model "
            "uses a naming scheme this client doesn't recognize."
        )

    def _preprocess_image(self, path: str | Path) -> np.ndarray:
        from PIL import Image

        pp = self.input_spec.get("ludwigPreprocessing") or {}
        height = pp.get("height", 224)
        width = pp.get("width", 224)

        img = Image.open(path).convert("RGB").resize((width, height))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0, 1]

        norm = self.input_spec.get("imageNormalization")
        if norm:
            mean = np.array(norm["mean"], dtype=np.float32)
            std = np.array(norm["std"], dtype=np.float32)
            arr = (arr - mean) / std
        # else: no known normalization preset — raw [0,1] pixels are passed
        # through. Check preprocessing.json's `ludwigPreprocessing` block if
        # predictions look off.

        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _preprocess_number(value: float, spec: dict) -> np.ndarray:
        """Replicates the exact normalization Ludwig fit at train time
        (ludwig.features.number_feature.numeric_transformation_registry) —
        the ONNX graph itself has none: NumberInputFeature.forward() goes
        straight from the raw tensor to its encoder.
        """
        x = float(value)
        norm = spec.get("numberNormalization")
        if norm:
            norm_type = norm.get("type")
            if norm_type == "zscore" and norm.get("std"):
                x = (x - norm["mean"]) / norm["std"]
            elif norm_type == "minmax" and norm.get("min") is not None and norm.get("max") is not None:
                span = norm["max"] - norm["min"]
                x = (x - norm["min"]) / span if span else 0.0
            elif norm_type == "log1p":
                x = math.log1p(x)
            elif norm_type == "iq" and norm.get("q2") is not None:
                span = (norm.get("q3") or 0) - (norm.get("q1") or 0)
                x = (x - norm["q2"]) / span if span else x - norm["q2"]
            # Unrecognized normalization type: passed through unchanged —
            # check preprocessing.json's `numberNormalization` if predictions look off.
        # 1-D, shape [1] — matches Ludwig's own create_sample_input() for a
        # single-example number feature (torch.rand([batch_size])).
        return np.array([x], dtype=np.float32)

    def _preprocess_record(self, record: dict[str, float]) -> dict[str, np.ndarray]:
        feed: dict[str, np.ndarray] = {}
        for spec in self.input_specs:
            if spec["type"] != "number":
                raise NotImplementedError(
                    f"No preprocessing implemented for input type '{spec['type']}' in a multi-input model. "
                    "See preprocessing.json and adapt this method."
                )
            if spec["column"] not in record:
                raise KeyError(f"Missing required field '{spec['column']}'")
            feed[self._onnx_input_names[spec["name"]]] = self._preprocess_number(record[spec["column"]], spec)
        return feed

    def predict(self, value: str | Path | dict[str, float], top_k: int = 5) -> dict[str, float]:
        """Run inference and return `{class_name: probability}` sorted
        descending (classification), or `{"value": float}` (regression).

        `value` is a file path for a single `image` input, or a
        `{column: value}` dict — one entry per input feature — for a
        tabular (`number`-only) model.
        """
        if self._is_tabular:
            if not isinstance(value, dict):
                raise TypeError("This model has tabular inputs — pass a dict of {column: value}, not a file path.")
            feed = self._preprocess_record(value)
        elif self.input_spec["type"] == "image":
            feed = {self._onnx_input_names[self.input_spec["name"]]: self._preprocess_image(value)}
        else:
            raise NotImplementedError(
                f"No preprocessing implemented for input type '{self.input_spec['type']}'. "
                "See preprocessing.json's `inputs[0].ludwigPreprocessing` for the "
                "raw parameters Ludwig used and adapt this method."
            )

        raw = self.session.run([self._onnx_output_name], feed)[0]
        logits = np.asarray(raw).reshape(-1)

        classes = self.output_spec.get("classes")
        if classes and len(logits) == len(classes):
            probs = _softmax(logits)
            ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
            return {name: round(float(p), 4) for name, p in ranked[:top_k]}

        return {"value": float(logits[0])}
