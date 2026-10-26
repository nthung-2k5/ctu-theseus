"""What a trained PyTorch model expects as input, written down once.

An `ImageSpec`, `TabularSpec` or `AudioSpec` is the single definition that training, evaluation, serving and the
export manifest all read, so the four cannot drift apart (the way a Ludwig model's resize and normalisation, which
Ludwig keeps to itself, can). It is stored in `names.json` beside the class order, and a loaded model rebuilds it from
there: it never consults the compiled config. Pure data (no torch, no PIL).
"""

import math
from dataclasses import dataclass, replace
from typing import Any


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
