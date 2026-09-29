"""Audio preprocessing. Samples are AudioClip (float32 numpy, shape channels x frames)."""

from typing import Any

from pydantic import Field

from theseus.augmentation.media import AudioClip
from theseus.preprocessing.base import NoParams, ParamsModel, Preprocessing


class Resample(Preprocessing):
    class Params(ParamsModel):
        sample_rate_hz: int = Field(16000, ge=4000, le=48000, title="Target sample rate (Hz)")

    id = "audio_resample"
    label = "Resample"
    description = "Resample every clip to a fixed sample rate."
    modality = "audio"
    order = 10

    @classmethod
    def apply(cls, sample: AudioClip, params: Params, state: Any = None) -> AudioClip:
        import numpy as np

        if sample.sample_rate == params.sample_rate_hz:
            return AudioClip(sample.samples.copy(), sample.sample_rate)
        frames = sample.samples.shape[1]
        new_frames = max(1, round(frames * params.sample_rate_hz / sample.sample_rate))
        positions = np.linspace(0, frames - 1, new_frames)
        source = np.arange(frames)
        resampled = np.stack([np.interp(positions, source, ch) for ch in sample.samples]).astype(np.float32)
        return AudioClip(resampled, params.sample_rate_hz)


class ToMono(Preprocessing):
    id = "audio_to_mono"
    label = "Mono"
    description = "Downmix multi-channel audio to a single channel by averaging."
    modality = "audio"
    order = 20

    @classmethod
    def apply(cls, sample: AudioClip, params: NoParams, state: Any = None) -> AudioClip:
        import numpy as np

        if sample.samples.shape[0] == 1:
            return AudioClip(sample.samples.copy(), sample.sample_rate)
        return AudioClip(np.mean(sample.samples, axis=0, keepdims=True).astype(np.float32), sample.sample_rate)
