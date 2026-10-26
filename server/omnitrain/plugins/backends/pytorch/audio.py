"""Audio file -> the log-mel spectrogram a PyTorch audio classifier takes: the one definition train, serve and evaluate
share, and the recipe the export manifest spells out for a client (`manifest.py`).

A clip is decoded, mixed down to mono by averaging its channels, and resampled to `spec.sample_rate` with
`scipy.signal.resample_poly` (up/down are the rates divided by their gcd). That whole waveform is what `load_waveform`
returns. A window of exactly `spec.length` samples is then taken from it (`window_starts`, `cut`): a clip that is too
short is zero-padded at the end, a longer one is cut (at a random place in training, at evenly spaced places when
scoring). The window becomes a spectrogram by `log_mel`: a Hann window (periodic) of `n_fft` samples every `hop`,
with no padding at the edges, the power spectrum `|X|^2` of each frame, an HTK mel filterbank (triangular, no area
normalisation) of `n_mels` bands from `f_min` to `sample_rate / 2`, then `ln(mel + 1e-6)`. `spectrogram` standardises
that with the spec's scalar `mean` and `std`. The result is `[n_mels, frames]`, and `tests/test_pytorch_backend.py`
checks it against torchaudio's own `MelSpectrogram`.

numpy and scipy only (torchaudio is imported inside `decode`, for the containers scipy cannot read): no torch.
"""

import functools
import math
import os

import numpy as np

from omnitrain.plugins.backends.pytorch.spec import AudioSpec

# Added to the mel energies before the log, so silence is a finite (very negative) number.
LOG_FLOOR = 1e-6


def decode(path: str) -> tuple[np.ndarray, int]:
    """A file's samples as float32 in [-1, 1], shape `(channels, frames)`, and its sample rate.

    WAV through scipy (no system codecs needed); any other container, and a WAV scipy rejects (ADPCM, 24-bit), through
    torchaudio."""
    if os.path.splitext(path)[1].lower() == ".wav":
        try:
            from scipy.io import wavfile

            rate, raw = wavfile.read(path)
            arr = np.asarray(raw)
            if arr.dtype == np.uint8:
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            else:
                arr = arr.astype(np.float32)
            arr = arr[None, :] if arr.ndim == 1 else arr.T
            return np.ascontiguousarray(arr), int(rate)
        except Exception:  # noqa: S110 - falls through to torchaudio below
            pass
    import torchaudio

    waveform, rate = torchaudio.load(path)
    return waveform.numpy().astype(np.float32), int(rate)


def to_mono_at(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """`(channels, frames)` audio at `rate` as one channel of float32 at `target_rate`."""
    mono = samples.mean(axis=0, dtype=np.float64) if samples.ndim == 2 else np.asarray(samples, dtype=np.float64)
    if rate != target_rate and len(mono):
        from scipy.signal import resample_poly

        divisor = math.gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)
    return mono.astype(np.float32)


def load_waveform(path: str, spec: AudioSpec) -> np.ndarray:
    """The whole clip at the model's rate, mono: what training caches and serving windows."""
    samples, rate = decode(path)
    return to_mono_at(samples, rate, spec.sample_rate)


def window_starts(total: int, length: int, count: int) -> list[int]:
    """Where to take `count` windows of `length` samples from a clip of `total`, for scoring. A clip no longer than
    one window gives one at 0 (zero-padded); a longer one gives windows spread evenly over it, one centred if `count`
    is 1."""
    if total <= length:
        return [0]
    room = total - length
    if count <= 1:
        return [room // 2]
    return sorted({round(i * room / (count - 1)) for i in range(count)})


def random_start(total: int, length: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, total - length + 1)) if total > length else 0


def cut(waveform: np.ndarray, start: int, length: int) -> np.ndarray:
    """`length` samples from `start`, zero-padded at the end when the clip runs out."""
    part = waveform[start : start + length]
    if len(part) < length:
        part = np.concatenate([part, np.zeros(length - len(part), dtype=np.float32)])
    return np.asarray(part, dtype=np.float32)


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


@functools.lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """`[n_fft // 2 + 1, n_mels]` triangular HTK filters, as `torchaudio.functional.melscale_fbanks(norm=None)`."""
    freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mels = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    edges = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    widths = np.diff(edges)
    slopes = edges[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    return np.maximum(0.0, np.minimum(down, up))


@functools.lru_cache(maxsize=8)
def _hann(n_fft: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)


def log_mel(window: np.ndarray, spec: AudioSpec) -> np.ndarray:
    """`[n_mels, frames]` float32 log-mel energies of one window of `spec.length` samples, before standardising."""
    frames = np.lib.stride_tricks.sliding_window_view(window.astype(np.float64), spec.n_fft)[:: spec.hop]
    power = np.abs(np.fft.rfft(frames * _hann(spec.n_fft), axis=1)) ** 2
    filters = mel_filterbank(spec.sample_rate, spec.n_fft, spec.n_mels, spec.f_min, spec.f_max)
    return np.log(power @ filters + LOG_FLOOR).T.astype(np.float32)


def spectrogram(window: np.ndarray, spec: AudioSpec) -> np.ndarray:
    """What the network is given: `log_mel` standardised with the spec's scalar mean and std."""
    return ((log_mel(window, spec) - spec.mean) / spec.std).astype(np.float32)


def spec_augment(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SpecAugment on a standardised spectrogram: two bands of frequencies and two stretches of time, each up to an
    eighth of its axis, set to 0 (the mean). Training only."""
    out = array.copy()
    bands, frames = out.shape
    for _ in range(2):
        width = int(rng.integers(0, max(bands // 8, 1) + 1))
        if width:
            start = int(rng.integers(0, bands - width + 1))
            out[start : start + width, :] = 0.0
        width = int(rng.integers(0, max(frames // 8, 1) + 1))
        if width:
            start = int(rng.integers(0, frames - width + 1))
            out[:, start : start + width] = 0.0
    return out
