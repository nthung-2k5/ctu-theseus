"""What a trained PyTorch model expects as input, written down once.

An `ImageSpec`, `TabularSpec` or `AudioSpec` is the single definition that training, evaluation, serving and the
export manifest all read, so the four cannot drift apart (the way a Ludwig model's resize and normalisation, which
Ludwig keeps to itself, can). It is stored in `names.json` beside the class order, and a loaded model rebuilds it from
there: it never consults the compiled config. Pure data (no torch, no PIL).
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Self


@dataclass(frozen=True)
class ImageSpec:
    # Side of the square the network sees. An image is resized so its SHORTER edge is `size` (bilinear), then the
    # centre `size` x `size` square is cut out: `imageResize.mode == "shorter_edge_center_crop"` in the manifest.
    size: int
    # Applied after scaling pixels to 0..1: (x - mean) / std per RGB channel.
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_meta(self) -> dict[str, Any]:
        return {"inputKind": "image", "imageSize": self.size, "mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "ImageSpec":
        mean, std = meta["mean"], meta["std"]
        return cls(
            size=int(meta["imageSize"]),
            mean=(float(mean[0]), float(mean[1]), float(mean[2])),
            std=(float(std[0]), float(std[1]), float(std[2])),
        )


@dataclass(frozen=True)
class TabularSpec:
    """A table row: one number per column, in this order. The network does its own preprocessing, INSIDE the graph
    (so the exported model takes raw values and the manifest asks a client to do nothing): a missing or non-finite
    value is replaced by the column's `fill`, then `(x - mean) / std`. All three are fitted on the TRAIN split only, so
    nothing about validation or test rows leaks into them. A regression target is standardised with `target_mean` and
    `target_std` the same way, and the graph un-does it so it outputs the target in its own units."""

    columns: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    fill: tuple[float, ...]
    target_mean: float = 0.0
    target_std: float = 1.0

    def to_meta(self) -> dict[str, Any]:
        return {
            "inputKind": "tabular",
            "inputColumns": list(self.columns),
            "mean": list(self.mean),
            "std": list(self.std),
            "fill": list(self.fill),
            "targetMean": self.target_mean,
            "targetStd": self.target_std,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "TabularSpec":
        return cls(
            columns=tuple(meta["inputColumns"]),
            mean=tuple(float(v) for v in meta["mean"]),
            std=tuple(float(v) for v in meta["std"]),
            fill=tuple(float(v) for v in meta["fill"]),
            target_mean=float(meta.get("targetMean", 0.0)),
            target_std=float(meta.get("targetStd", 1.0)),
        )


def safe_std(std: float) -> float:
    """A column that never varies (or has one row) would divide by zero; standardising it gives all zeros instead."""
    return std if math.isfinite(std) and std > 1e-12 else 1.0


@dataclass(frozen=True)
class AudioSpec:
    """An audio clip as the network sees it: a log-mel spectrogram of a fixed-length window. `audio.py` is the one
    recipe (decode, mix to mono, resample to `sample_rate`, cut or zero-pad to `seconds`, Hann-windowed STFT of `n_fft`
    samples every `hop`, power spectrum, HTK mel filterbank of `n_mels` bands from `f_min` to `f_max`, natural log of
    `mel + 1e-6`, then `(x - mean) / std`), used by training, evaluation and serving, and written out in full in the
    export manifest for a client to follow. `mean` and `std` are scalars fitted on the TRAIN clips only.

    A clip longer than `seconds` is scored as `clips` windows spread evenly over it, their probabilities averaged."""

    sample_rate: int
    seconds: float
    n_mels: int
    n_fft: int = 1024
    hop: int = 256
    f_min: float = 0.0
    mean: float = 0.0
    std: float = 1.0
    clips: int = 3

    @property
    def f_max(self) -> float:
        return self.sample_rate / 2

    @property
    def length(self) -> int:
        """Samples in one window."""
        return round(self.seconds * self.sample_rate)

    @property
    def frames(self) -> int:
        """Spectrogram columns of one window (no padding at the edges, so only whole frames count)."""
        return 1 + (self.length - self.n_fft) // self.hop

    def with_normalization(self, mean: float, std: float) -> "AudioSpec":
        return replace(self, mean=mean, std=safe_std(std))

    def to_meta(self) -> dict[str, Any]:
        return {
            "inputKind": "audio",
            "sampleRate": self.sample_rate,
            "clipSeconds": self.seconds,
            "nMels": self.n_mels,
            "nFft": self.n_fft,
            "hopLength": self.hop,
            "fMin": self.f_min,
            "spectrogramMean": self.mean,
            "spectrogramStd": self.std,
            "clipsPerAudio": self.clips,
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "AudioSpec":
        return cls(
            sample_rate=int(meta["sampleRate"]),
            seconds=float(meta["clipSeconds"]),
            n_mels=int(meta["nMels"]),
            n_fft=int(meta["nFft"]),
            hop=int(meta["hopLength"]),
            f_min=float(meta["fMin"]),
            mean=float(meta["spectrogramMean"]),
            std=float(meta["spectrogramStd"]),
            clips=int(meta["clipsPerAudio"]),
        )


@dataclass(frozen=True)
class TextSpec:
    """A text as the network sees it: `max_tokens` ids. `text.py` is the one recipe (lower-case ASCII letters, split on
    separators, look each token up in `vocab`, `UNKNOWN` when absent, cut at `max_tokens`, pad with `PAD`), used by
    training, evaluation and serving, and written out in full in the export manifest for a client to follow. The
    vocabulary is built from the TRAIN texts only. A word's id is its position in `vocab` plus two (0 is padding, 1 is
    an unknown word)."""

    max_tokens: int
    vocab: tuple[str, ...] = ()

    @property
    def index(self) -> dict[str, int]:
        return {word: i + 2 for i, word in enumerate(self.vocab)}

    @property
    def size(self) -> int:
        """Rows of the embedding table: the padding and unknown ids plus the vocabulary."""
        return len(self.vocab) + 2

    def with_vocabulary(self, vocab: list[str]) -> Self:
        return replace(self, vocab=tuple(vocab))

    def to_meta(self) -> dict[str, Any]:
        return {"inputKind": "text", "maxTokens": self.max_tokens, "vocabulary": list(self.vocab)}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> Self:
        return cls(max_tokens=int(meta["maxTokens"]), vocab=tuple(str(w) for w in meta.get("vocabulary", ())))


@dataclass(frozen=True)
class WordsSpec(TextSpec):
    """A text as the tagger sees it: one id per whitespace-separated word (`tagging.py`), in the same vocabulary layout
    as a `TextSpec` (0 padding, 1 unknown, vocabulary position plus two) but a different recipe, so another
    `inputKind`: a word is never split further, and the model gives one tag per word."""

    def to_meta(self) -> dict[str, Any]:
        return {**super().to_meta(), "inputKind": "words"}


@dataclass(frozen=True)
class HFTextSpec:
    """A text as a bring-your-own Hugging Face model sees it: `max_tokens` ids from the model's own tokenizer. The
    tokenizer is a Hugging Face `tokenizers` definition (`tokenizer`, the JSON string that library writes), which
    already holds the normaliser, the splitting rule, the vocabulary and the special tokens it adds ([CLS] and [SEP], or
    <s> and </s>). A text is encoded with that, cut at `max_tokens` (the special tokens count), and padded at the end
    with `pad_id` (the token `pad_token`) up to `max_tokens`. The network takes only the ids: it derives its attention
    mask from `pad_id`, so no real token may carry it."""

    max_tokens: int
    pad_id: int
    pad_token: str
    tokenizer: str

    def to_meta(self) -> dict[str, Any]:
        return {
            "inputKind": "text",
            "hfTokenizer": {
                "maxTokens": self.max_tokens,
                "padId": self.pad_id,
                "padToken": self.pad_token,
                "tokenizer": self.tokenizer,
            },
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "HFTextSpec":
        found = meta["hfTokenizer"]
        return cls(
            max_tokens=int(found["maxTokens"]),
            pad_id=int(found["padId"]),
            pad_token=str(found["padToken"]),
            tokenizer=str(found["tokenizer"]),
        )
