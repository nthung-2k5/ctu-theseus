"""OmniTrain-generated inference client.

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
  - a `detections` output (object detection, YOLO) — pass an image path;
    `predict()` returns a list of `{label, confidence, box}` dicts, `box`
    being `{x, y, w, h}`: the top-left corner and size as fractions (0..1) of
    the image. The ONNX graph returns every candidate box; this client does the
    confidence filter, per-class non-maximum suppression and the undoing of the
    letterbox.
  - a `video` input (video classification) — pass a video file path. Needs
    `opencv-python-headless` to decode it.
  - an `audio` input with an `audioSpectrogram` recipe (audio classification) — pass an audio file
    path. A WAV file is read here; other containers need the optional `soundfile` package. The
    spectrogram is computed with numpy alone, from the recipe in preprocessing.json.

  - a `text` input with a `textTokens` recipe (text classification) — pass the text itself, a
    `str`, not a file path. It is split into words and looked up in the vocabulary that ships in
    preprocessing.json, with the standard library alone.
  - a `text` input with an `hfTokenizer` recipe (a fine-tuned Hugging Face transformer) — pass the
    text itself. It is tokenized by the model's own tokenizer, which ships in the bundle as
    `tokenizer.json` (or inside preprocessing.json) and is run by the `tokenizers` package.
  - a `text` input with a `tagWords` recipe and a `tags` output (token classification) — pass the
    text itself; `predict()` returns one `{token, tag, confidence}` per whitespace-separated word.
  - a `text` input with an `hfTokenizer` recipe `hf_word_tokens_v1` and a `tags` output (a fine-tuned
    Hugging Face token classifier) — the same call and result; the model's own tokenizer reads the
    words, and a word is tagged by the first token that covers it.

Every other input type (Ludwig's text and audio, a scikit-learn text model) raises
NotImplementedError with a pointer to preprocessing.json so you can adapt preprocessing yourself —
text tokenization in particular depends on the chosen encoder's own vocabulary (Ludwig's built-in
tokenizer, or a HuggingFace subword tokenizer for bert/distilbert/roberta), which can't be
generically reconstructed from preprocessing.json alone. See README.md.
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort

_HERE = Path(__file__).parent


def _first_tokens(offsets: list[tuple[int, int]], words: list[str]) -> list[int]:
    """For each word, the position of the first token that covers it, or -1. `offsets` are the character spans of the
    tokens in the words joined by single spaces (special tokens and padding have the span (0, 0)); a token belongs to
    the word its first non-space character is in."""
    owner: list[int] = []
    for i, word in enumerate(words):
        if i:
            owner.append(-1)
        owner.extend([i] * len(word))
    first = [-1] * len(words)
    for position, (start, end) in enumerate(offsets):
        for char in range(start, min(end, len(owner))):
            if owner[char] >= 0:
                if first[owner[char]] < 0:
                    first[owner[char]] = position
                break
    return first


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _shorter_edge_geometry(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    """(resized_width, resized_height, crop_left, crop_top) for `imageResize.mode == "shorter_edge_center_crop"`:
    scale so the SHORTER edge is `size` (the other edge is truncated to a whole pixel), then centre-crop a
    size x size square. The same arithmetic as torchvision's Resize(size) followed by CenterCrop(size), including
    its round-half-to-even for an odd leftover."""
    if width <= height:
        new_width, new_height = size, int(size * height / width)
    else:
        new_width, new_height = int(size * width / height), size
    return new_width, new_height, round((new_width - size) / 2.0), round((new_height - size) / 2.0)


# -- Audio ---------------------------------------------------------------------------------------------------------
# `inputs[0].audioSpectrogram` (recipe "log_mel_v1") spelled out in numpy. Every number comes from the manifest.


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """A WAV file's samples as float32 `(channels, frames)` in [-1, 1], and its sample rate. Integer PCM of 8, 16, 24
    and 32 bits and 32/64-bit float, including WAVE_FORMAT_EXTENSIBLE. Anything else raises ValueError."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt = None
    pcm = b""
    position = 12
    while position + 8 <= len(data):
        chunk, size = data[position : position + 4], struct.unpack_from("<I", data, position + 4)[0]
        body = position + 8
        if chunk == b"fmt " and size >= 16:
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if tag == 0xFFFE and size >= 26:  # extensible: the real format tag is the sub-format's first two bytes
                tag = struct.unpack_from("<H", data, body + 24)[0]
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            pcm = data[body : body + size]  # a streamed file may claim more than it holds
            break
        position = body + size + (size & 1)
    if fmt is None or not pcm:
        raise ValueError("the WAV file has no format or no samples")
    tag, channels, rate, bits = fmt
    if channels < 1:
        raise ValueError("the WAV file has no channels")
    width = bits // 8
    pcm = pcm[: len(pcm) // (width * channels) * (width * channels)]
    if tag == 1 and bits == 8:
        values = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    elif tag == 1 and bits == 24:
        raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(signed >= 1 << 23, signed - (1 << 24), signed).astype(np.float32) / 8388607.0
    elif tag == 1 and bits == 32:
        values = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483647.0
    elif tag == 3 and bits in (32, 64):
        values = np.frombuffer(pcm, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV encoding (format {tag}, {bits} bits)")
    return np.ascontiguousarray(values.reshape(-1, channels).T), int(rate)


def _decode_audio(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        return _read_wav(Path(path).read_bytes())
    except ValueError as wav_error:
        try:
            import soundfile
        except ImportError:
            raise ValueError(
                f"Could not read {path} as a WAV file ({wav_error}). Other audio formats need the optional "
                "`soundfile` package (pip install soundfile)."
            ) from None
        samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(samples.T), int(rate)


def _resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    """`scipy.signal.resample_poly(x, up, down)` without scipy: the polyphase filter in `audioSpectrogram.resample`."""
    divisor = math.gcd(rate, target)
    up, down = target // divisor, rate // divisor
    big = max(up, down)
    half = 10 * big
    n_out = -(-len(x) * up // down)
    taps = np.arange(2 * half + 1) - half
    h = np.sinc(taps / big) / big * np.kaiser(2 * half + 1, 5.0)
    h *= up / h.sum()
    padded = np.concatenate([np.zeros(half // up + 2), x, np.zeros(half // up + 2)])
    pad = half // up + 2
    out = np.zeros(n_out)
    index = np.arange(n_out)
    for phase in range(up):
        # Outputs i whose position i * down lies `phase` samples past a multiple of `up`.
        mine = index[(index * down) % up == phase]
        if not mine.size:
            continue
        first = -((half - phase) // up)  # smallest d with half - phase + d * up >= 0
        last = (half + phase) // up  # largest d with half - phase + d * up <= 2 * half
        d = np.arange(first, last + 1)
        taps_d = h[half - phase + d * up]
        starts = mine * down // up + first + pad
        view = np.lib.stride_tricks.sliding_window_view(padded, len(d))
        for chunk in range(0, len(mine), 32768):  # bounded memory for a long recording
            out[mine[chunk : chunk + 32768]] = view[starts[chunk : chunk + 32768]] @ taps_d
    return out


def _window_starts(total: int, length: int, count: int) -> list[int]:
    if total <= length:
        return [0]
    room = total - length
    if count <= 1:
        return [room // 2]
    return sorted({round(i * room / (count - 1)) for i in range(count)})  # Python's round() is half-to-even


def _mel_filterbank(sample_rate: int, n_fft: int, bands: int, f_min: float, f_max: float) -> np.ndarray:
    """`[n_fft // 2 + 1, bands]` triangular HTK filters."""
    freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mels = np.linspace(2595.0 * np.log10(1.0 + f_min / 700.0), 2595.0 * np.log10(1.0 + f_max / 700.0), bands + 2)
    edges = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    slopes = edges[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / np.diff(edges)[:-1]
    up = slopes[:, 2:] / np.diff(edges)[1:]
    return np.maximum(0.0, np.minimum(down, up))


def _text_splitter(recipe: dict) -> tuple[dict[int, int], re.Pattern[str]]:
    """(the lower-casing table, the pattern a token is cut at), both built from the recipe's own numbers."""
    lower = recipe["lowercase"]
    table = {code: code + lower["add"] for code in range(lower["from"], lower["to"] + 1)}
    ranges = "".join(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}" for lo, hi in recipe["separatorRanges"])
    return table, re.compile(f"[{ranges}]+")


def _non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float
) -> list[int]:
    """Indices of the boxes to keep, most confident first: within each class, a box is dropped when it overlaps a
    more confident one by more than `iou_threshold` (intersection over union). Boxes are `[x1, y1, x2, y2]`."""
    kept: list[int] = []
    for label in np.unique(classes):
        indices = np.flatnonzero(classes == label)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        while order.size:
            first, rest = order[0], order[1:]
            kept.append(int(first))
            if not rest.size:
                break
            x1 = np.maximum(boxes[first, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[first, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[first, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[first, 3], boxes[rest, 3])
            intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area_first = (boxes[first, 2] - boxes[first, 0]) * (boxes[first, 3] - boxes[first, 1])
            area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            overlap = intersection / np.maximum(area_first + area_rest - intersection, 1e-9)
            order = rest[overlap <= iou_threshold]
    kept.sort(key=lambda i: -scores[i])
    return kept


class OmniTrainClient:
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
        # What `predict()` does with its input: "detection" (an image in, a list of boxes out), "video" or "audio" (a
        # file in, class probabilities out) or "standard" (everything else this client handles).
        if self.output_spec["type"] == "detections":
            self._kind = "detection"
        elif self.input_spec["type"] == "video":
            self._kind = "video"
        elif self.input_spec["type"] == "audio" and "audioSpectrogram" in self.input_spec:
            self._kind = "audio"
            self._audio_recipe = self.input_spec["audioSpectrogram"]
            if self._audio_recipe.get("recipe") != "log_mel_v1":
                raise NotImplementedError(
                    f"Unknown audio recipe {self._audio_recipe.get('recipe')!r}. "
                    "See preprocessing.json's `inputs[0].audioSpectrogram` and adapt this client."
                )
        elif self.output_spec["type"] == "tags" and "tagWords" in self.input_spec:
            self._kind = "tags"
            self._tag_recipe = self.input_spec["tagWords"]
            if self._tag_recipe.get("recipe") != "tag_word_ids_v1":
                raise NotImplementedError(
                    f"Unknown tagging recipe {self._tag_recipe.get('recipe')!r}. "
                    "See preprocessing.json's `inputs[0].tagWords` and adapt this client."
                )
            lower = self._tag_recipe["lowercase"]
            self._tag_lower = {code: code + lower["add"] for code in range(lower["from"], lower["to"] + 1)}
            first = self._tag_recipe["firstWordId"]
            self._tag_ids = {word: first + i for i, word in enumerate(self._tag_recipe["vocabulary"])}
        elif self.output_spec["type"] == "tags" and "hfTokenizer" in self.input_spec:
            self._kind = "tags"
            self._tag_recipe = self.input_spec["hfTokenizer"]
            if self._tag_recipe.get("recipe") != "hf_word_tokens_v1":
                raise NotImplementedError(
                    f"Unknown tagging recipe {self._tag_recipe.get('recipe')!r}. "
                    "See preprocessing.json's `inputs[0].hfTokenizer` and adapt this client."
                )
            self._hf_tokenizer = self._load_hf_tokenizer(self._tag_recipe, Path(preprocessing_path).parent)
        elif self.input_spec["type"] == "text" and "textTokens" in self.input_spec:
            self._kind = "text"
            self._text_recipe = self.input_spec["textTokens"]
            if self._text_recipe.get("recipe") != "word_ids_v1":
                raise NotImplementedError(
                    f"Unknown text recipe {self._text_recipe.get('recipe')!r}. "
                    "See preprocessing.json's `inputs[0].textTokens` and adapt this client."
                )
            self._text_lower, self._text_split = _text_splitter(self._text_recipe)
            first = self._text_recipe["firstWordId"]
            self._text_ids = {word: first + i for i, word in enumerate(self._text_recipe["vocabulary"])}
        elif self.input_spec["type"] == "text" and "hfTokenizer" in self.input_spec:
            self._kind = "text"
            self._text_recipe = self.input_spec["hfTokenizer"]
            if self._text_recipe.get("recipe") != "hf_tokenizer_v1":
                raise NotImplementedError(
                    f"Unknown text recipe {self._text_recipe.get('recipe')!r}. "
                    "See preprocessing.json's `inputs[0].hfTokenizer` and adapt this client."
                )
            self._hf_tokenizer = self._load_hf_tokenizer(self._text_recipe, Path(preprocessing_path).parent)
        else:
            self._kind = "standard"

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

        img = Image.open(path).convert("RGB")
        mode = (self.input_spec.get("imageResize") or {}).get("mode")
        if mode is None:
            img = img.resize((width, height))  # the whole image squashed to the target size
        elif mode == "shorter_edge_center_crop":
            size = min(width, height)
            new_width, new_height, left, top = _shorter_edge_geometry(img.width, img.height, size)
            img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
            img = img.crop((left, top, left + width, top + height))
        else:
            raise NotImplementedError(
                f"Unsupported imageResize mode {mode!r}. "
                "See preprocessing.json's `inputs[0].imageResize` and adapt this client."
            )
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

    def predict(self, value: str | Path | dict[str, float], top_k: int = 5) -> dict[str, float] | list[dict]:
        """Run inference and return `{class_name: probability}` sorted
        descending (classification), or `{"value": float}` (regression), or,
        for a detector, the list `detect()` returns, or, for a tagger, one
        `{token, tag, confidence}` per word of the text.

        `value` is a file path for a single `image`, `video` or `audio` input, the text
        itself (a `str`) for a `text` input, or a `{column: value}` dict — one entry per
        input feature — for a tabular (`number`-only) model.
        """
        if self._kind == "detection":
            return self.detect(value)
        if self._kind == "video":
            return self._classify_video(value, top_k)
        if self._kind == "audio":
            return self._classify_audio(value, top_k)
        if self._kind == "tags":
            return self._tag_words(str(value))
        if self._kind == "text":
            return self._classify_text(str(value), top_k)
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

    # -- Detection -------------------------------------------------------------------------------------------------

    def _letterbox(self, path: str | Path) -> tuple[np.ndarray, float, int, int, int, int]:
        """The image as the detector's input tensor, and what is needed to map its boxes back.

        Returns (tensor, scale, pad_left, pad_top, original_width, original_height). `imageResize.mode == "letterbox"`:
        scale so the LONGER edge is `size` (bilinear), pad the shorter one, centred, with `padValue` grey up to a
        size x size square (the arithmetic of ultralytics.data.augment.LetterBox). RGB, 0..1, no mean/std.
        """
        from PIL import Image

        resize = self.input_spec.get("imageResize") or {}
        if resize.get("mode") != "letterbox":
            raise NotImplementedError(
                f"Unsupported imageResize mode {resize.get('mode')!r} for a detector. "
                "See preprocessing.json's `inputs[0].imageResize` and adapt this client."
            )
        height, width = resize["size"]
        pad_value = resize.get("padValue", 114)

        img = Image.open(path).convert("RGB")
        original_width, original_height = img.size
        scale = min(height / original_height, width / original_width)
        new_width, new_height = round(original_width * scale), round(original_height * scale)
        if (new_width, new_height) != (original_width, original_height):
            img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        dw, dh = (width - new_width) / 2, (height - new_height) / 2
        left, top = round(dw - 0.1), round(dh - 0.1)

        canvas = np.full((height, width, 3), pad_value, dtype=np.uint8)
        canvas[top : top + new_height, left : left + new_width] = np.asarray(img, dtype=np.uint8)
        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        return tensor, scale, left, top, original_width, original_height

    def detect(
        self,
        path: str | Path,
        confidence: float | None = None,
        iou: float | None = None,
        max_detections: int = 300,
    ) -> list[dict]:
        """Boxes found in an image, most confident first: `[{"label", "confidence", "box": {x, y, w, h}}]`.

        `box` is the top-left corner and the size, as fractions (0..1) of the ORIGINAL image. `confidence` and `iou`
        default to the thresholds in preprocessing.json (`outputs[0].yoloOutput`).
        """
        info = self.output_spec.get("yoloOutput") or {}
        confidence = info.get("confidenceThreshold", 0.25) if confidence is None else confidence
        iou = info.get("iouThreshold", 0.7) if iou is None else iou
        classes = self.output_spec["classes"]

        tensor, scale, left, top, original_width, original_height = self._letterbox(path)
        raw = self.session.run([self._onnx_output_name], {self._onnx_input_names[self.input_spec["name"]]: tensor})[0]
        # [1, 4 + classes, candidates] -> one row per candidate: centre x, centre y, width, height, then a score per class.
        candidates = np.asarray(raw)[0].T
        scores = candidates[:, 4:]
        best_class = scores.argmax(axis=1)
        best_score = scores.max(axis=1)
        keep = best_score >= confidence
        candidates, best_class, best_score = candidates[keep], best_class[keep], best_score[keep]

        cx, cy, w, h = candidates[:, 0], candidates[:, 1], candidates[:, 2], candidates[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        detections: list[dict] = []
        for index in _non_max_suppression(boxes, best_score, best_class, iou)[:max_detections]:
            # Undo the letterbox: remove the padding, divide by the scale, clip to the image.
            x1, y1, x2, y2 = boxes[index]
            x1, x2 = ((np.clip([x1, x2], 0, None) - left) / scale).clip(0, original_width)
            y1, y2 = ((np.clip([y1, y2], 0, None) - top) / scale).clip(0, original_height)
            detections.append(
                {
                    "label": classes[int(best_class[index])],
                    "confidence": round(float(best_score[index]), 4),
                    "box": {
                        "x": round(float(x1 / original_width), 6),
                        "y": round(float(y1 / original_height), 6),
                        "w": round(float((x2 - x1) / original_width), 6),
                        "h": round(float((y2 - y1) / original_height), 6),
                    },
                }
            )
        return detections

    # -- Video -----------------------------------------------------------------------------------------------------

    def _video_clips(self, path: str | Path) -> np.ndarray:
        """The video as a batch of clips, `[clips, 3, frames, size, size]` float32.

        Follows `inputs[0].videoSampling`: `clipsPerVideo` clips of `numFrames` frames taken `frameStride` source
        frames apart, the clips' starts spread evenly over the video (one centred clip if there is one), the last
        frame repeated for a video too short to fill a clip. Each frame is RGB, resized so its shorter edge is `size`
        (bilinear, OpenCV), centre-cropped to a square, scaled to 0..1 and normalised with `imageNormalization`.
        """
        import cv2

        sampling = self.input_spec["videoSampling"]
        num_frames, stride, clips = sampling["numFrames"], sampling["frameStride"], sampling["clipsPerVideo"]
        size = self.input_spec["ludwigPreprocessing"]["height"]
        span = (num_frames - 1) * stride + 1

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open the video {path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError("The video has no frames")
            room = max(count - span, 0)
            if clips <= 1 or room == 0:
                starts = [room // 2] if clips <= 1 else [0] * clips
            else:
                starts = [round(room * i / (clips - 1)) for i in range(clips)]
            last = count - 1
            plans = [[min(start + i * stride, last) for i in range(num_frames)] for start in starts]

            # Decode forward once: seeking in a compressed video lands on the nearest keyframe, not the frame asked for.
            wanted = {n for plan in plans for n in plan}
            decoded: dict[int, np.ndarray] = {}
            for number in range(max(wanted) + 1):
                if not capture.grab():
                    break
                if number in wanted:
                    ok, bgr = capture.retrieve()
                    if ok:
                        decoded[number] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        if not decoded:
            raise ValueError("The video has no decodable frames")

        norm = self.input_spec["imageNormalization"]
        mean = np.array(norm["mean"], dtype=np.float32).reshape(3, 1, 1, 1)
        std = np.array(norm["std"], dtype=np.float32).reshape(3, 1, 1, 1)
        batch = []
        for plan in plans:
            # A frame the decoder did not deliver is replaced by the nearest earlier one that arrived, or the first.
            frames, previous = [], decoded[min(decoded)]
            for number in plan:
                previous = decoded.get(number, previous)
                frames.append(previous)
            resized = []
            for frame in frames:
                height, width = frame.shape[:2]
                factor = size / min(height, width)
                new_width, new_height = max(size, round(width * factor)), max(size, round(height * factor))
                if (new_width, new_height) != (width, height):
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                resized.append(frame)
            top, left = (resized[0].shape[0] - size) // 2, (resized[0].shape[1] - size) // 2
            clip = np.stack([f[top : top + size, left : left + size] for f in resized])  # T x size x size x 3
            tensor = np.ascontiguousarray(clip).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
            batch.append((tensor - mean) / std)
        return np.stack(batch).astype(np.float32)

    def _classify_video(self, path: str | Path, top_k: int) -> dict[str, float]:
        """Class probabilities of a video: each clip's logits through a softmax, averaged over the clips."""
        clips = self._video_clips(path)
        raw = self.session.run([self._onnx_output_name], {self._onnx_input_names[self.input_spec["name"]]: clips})[0]
        logits = np.asarray(raw).reshape(len(clips), -1)
        probs = np.mean([_softmax(row) for row in logits], axis=0)
        classes = self.output_spec["classes"]
        ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
        return {name: round(float(p), 4) for name, p in ranked[:top_k]}

    # -- Audio -----------------------------------------------------------------------------------------------------

    def _audio_spectrograms(self, path: str | Path) -> np.ndarray:
        """The recording as a batch of spectrograms, `[windows, 1, bands, frames]` float32, by `audioSpectrogram`:
        decode, average the channels, resample, take `windowsPerAudio` windows (the last zero-padded if the recording
        is shorter), then for each a Hann-windowed power spectrum, mel filterbank, `ln(x + logFloor)` and
        `(x - mean) / std`."""
        recipe = self._audio_recipe
        stft, mel, norm = recipe["stft"], recipe["mel"], recipe["normalization"]
        samples, rate = _decode_audio(path)
        mono = samples.mean(axis=0, dtype=np.float64)
        if not len(mono):
            raise ValueError("The recording has no samples")
        wave = _resample(mono, rate, recipe["sampleRate"]).astype(np.float32)
        length = recipe["windowSamples"]
        n_fft, hop = stft["nFft"], stft["hopLength"]
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
        filters = _mel_filterbank(recipe["sampleRate"], n_fft, mel["bands"], mel["fMin"], mel["fMax"])
        batch = []
        for start in _window_starts(len(wave), length, recipe["windowsPerAudio"]):
            window = wave[start : start + length]
            if len(window) < length:
                window = np.concatenate([window, np.zeros(length - len(window), dtype=np.float32)])
            frames = np.lib.stride_tricks.sliding_window_view(window.astype(np.float64), n_fft)[::hop]
            power = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2
            log_mel = np.log(power @ filters + recipe["logFloor"]).T.astype(np.float32)
            batch.append(((log_mel - norm["mean"]) / norm["std"]).astype(np.float32))
        return np.stack(batch)[:, np.newaxis]

    def _classify_audio(self, path: str | Path, top_k: int) -> dict[str, float]:
        """Class probabilities of a recording: each window's logits through a softmax, averaged over the windows."""
        spectrograms = self._audio_spectrograms(path)
        feed = {self._onnx_input_names[self.input_spec["name"]]: spectrograms}
        raw = self.session.run([self._onnx_output_name], feed)[0]
        logits = np.asarray(raw).reshape(len(spectrograms), -1)
        probs = np.mean([_softmax(row) for row in logits], axis=0)
        classes = self.output_spec["classes"]
        ranked = sorted(zip(classes, probs), key=lambda p: p[1], reverse=True)
        return {name: round(float(p), 4) for name, p in ranked[:top_k]}

    # -- Tags ------------------------------------------------------------------------------------------------------

    def _word_ids(self, words: list[str]) -> np.ndarray:
        """`[1, maxTokens]` int64 ids by `tagWords`: each of the first maxTokens words, ASCII capitals lower-cased,
        looked up exactly in the vocabulary; the end is padded."""
        recipe = self._tag_recipe
        ids = [self._tag_ids.get(w.translate(self._tag_lower), recipe["unknownId"]) for w in words[: recipe["maxTokens"]]]
        ids += [recipe["padId"]] * (recipe["maxTokens"] - len(ids))
        return np.array([ids], dtype=np.int64)

    def _tag_words(self, text: str) -> list[dict]:
        """One `{token, tag, confidence}` per word of `text` (`str.split()`): the highest-scoring tag of the word's
        row of logits and its softmax probability. A word past `maxTokens` was not read and gets the recipe's
        `fallbackTag` with confidence 0."""
        recipe = self._tag_recipe
        words = text.split()
        if not words:
            return []
        if "tagWords" in self.input_spec:
            ids = self._word_ids(words)
            # The word's own row of logits: one word per position, the ones past maxTokens never read.
            row_of_word = [i if i < recipe["maxTokens"] else -1 for i in range(len(words))]
        else:
            # A Hugging Face tokenizer reads the words joined by single spaces; its offsets say which token is
            # the first of each word, and that token's row of logits is the word's.
            encoding = self._hf_tokenizer.encode(" ".join(words))
            ids = np.array([encoding.ids], dtype=np.int64)
            row_of_word = _first_tokens(encoding.offsets, words)
        feed = {self._onnx_input_names[self.input_spec["name"]]: ids}
        raw = np.asarray(self.session.run([self._onnx_output_name], feed)[0])
        rows = raw.reshape(raw.shape[-2], raw.shape[-1])
        classes = self.output_spec["classes"]
        out = []
        for word, row in zip(words, row_of_word):
            if row >= 0:
                probs = _softmax(rows[row])
                best = int(np.argmax(probs))
                out.append({"token": word, "tag": classes[best], "confidence": round(float(probs[best]), 4)})
            else:
                out.append({"token": word, "tag": recipe["fallbackTag"], "confidence": 0.0})
        return out

    # -- Text ------------------------------------------------------------------------------------------------------

    @staticmethod
    def _load_hf_tokenizer(recipe: dict, folder: Path):
        """The model's own Hugging Face tokenizer, set to cut at `maxTokens` and pad to it. It is stored whole in the
        recipe (`tokenizer`) or, in a bundle, in the file the recipe names (`tokenizerFile`, `tokenizer.json`)."""
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError("This model's text recipe needs the tokenizers package: pip install tokenizers") from e
        if "tokenizer" in recipe:
            tokenizer = Tokenizer.from_str(recipe["tokenizer"])
        else:
            tokenizer = Tokenizer.from_file(str(folder / recipe["tokenizerFile"]))
        tokenizer.enable_truncation(recipe["maxTokens"])
        tokenizer.enable_padding(length=recipe["maxTokens"], pad_id=recipe["padId"], pad_token=recipe["padToken"])
        return tokenizer

    def _text_tokens(self, text: str) -> np.ndarray:
        """`[1, maxTokens]` int64 ids: a word-id model looks the first maxTokens words up in its vocabulary, a
        Hugging Face model runs its own tokenizer; either way the end is padded."""
        recipe = self._text_recipe
        if "hfTokenizer" in self.input_spec:
            return np.array([self._hf_tokenizer.encode(text).ids], dtype=np.int64)
        tokens = [t for t in self._text_split.split(text.translate(self._text_lower)) if t]
        ids = [self._text_ids.get(t, recipe["unknownId"]) for t in tokens[: recipe["maxTokens"]]]
        ids += [recipe["padId"]] * (recipe["maxTokens"] - len(ids))
        return np.array([ids], dtype=np.int64)

    def _classify_text(self, text: str, top_k: int) -> dict[str, float]:
        """Class probabilities of a text: the logits of its one row through a softmax."""
        feed = {self._onnx_input_names[self.input_spec["name"]]: self._text_tokens(text)}
        logits = np.asarray(self.session.run([self._onnx_output_name], feed)[0]).reshape(-1)
        classes = self.output_spec["classes"]
        ranked = sorted(zip(classes, _softmax(logits)), key=lambda p: p[1], reverse=True)
        return {name: round(float(p), 4) for name, p in ranked[:top_k]}
