"""Audio augmentations. Samples are AudioClip (float32 numpy, shape channels x frames)."""

import random
from typing import Any

from pydantic import Field

from theseus.augmentation.base import Augmentation, ParamsModel
from theseus.augmentation.media import AudioClip


class Gain(Augmentation):
    class Params(ParamsModel):
        max_db: float = Field(6.0, ge=1.0, le=20.0, title="Max gain (dB)", description="Louder or quieter, up to this.")

    id = "audio_gain"
    label = "Volume change"
    description = "Make the clip randomly louder or quieter. Peaks are clipped, never wrapped."
    modality = "audio"
    order = 10

    @classmethod
    def apply(cls, sample: AudioClip, params: Params, rng: random.Random, state: Any = None) -> AudioClip:
        import numpy as np

        db = rng.uniform(-params.max_db, params.max_db)
        return AudioClip(np.clip(sample.samples * (10.0 ** (db / 20.0)), -1.0, 1.0), sample.sample_rate)


class AddNoise(Augmentation):
    class Params(ParamsModel):
        snr_db: float = Field(
            20.0, ge=5.0, le=40.0, title="Signal-to-noise (dB)", description="Lower is noisier.",
            json_schema_extra={"step": 1},
        )  # fmt: skip

    id = "audio_noise"
    label = "Background noise"
    description = "Mix in white noise at the given signal-to-noise ratio."
    modality = "audio"
    order = 20

    @classmethod
    def apply(cls, sample: AudioClip, params: Params, rng: random.Random, state: Any = None) -> AudioClip:
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(sample.samples))))
        if rms == 0.0:
            return sample  # silence: there is no signal level to set a ratio against
        noise_rms = rms / (10.0 ** (params.snr_db / 20.0))
        gen = np.random.default_rng(rng.getrandbits(64))
        noise = gen.normal(0.0, noise_rms, sample.samples.shape).astype(np.float32)
        return AudioClip(np.clip(sample.samples + noise, -1.0, 1.0), sample.sample_rate)


class TimeShift(Augmentation):
    class Params(ParamsModel):
        max_shift_seconds: float = Field(
            0.3, ge=0.05, le=2.0, title="Max shift (seconds)", json_schema_extra={"step": 0.05}
        )

    id = "audio_time_shift"
    label = "Time shift"
    description = "Slide the clip earlier or later; what runs off one end wraps around to the other."
    modality = "audio"
    order = 30

    @classmethod
    def apply(cls, sample: AudioClip, params: Params, rng: random.Random, state: Any = None) -> AudioClip:
        import numpy as np

        frames = sample.samples.shape[1]
        limit = min(int(params.max_shift_seconds * sample.sample_rate), max(frames - 1, 0))
        shift = rng.randint(-limit, limit) if limit else 0
        return AudioClip(np.roll(sample.samples, shift, axis=1), sample.sample_rate)


class SpeedPerturb(Augmentation):
    class Params(ParamsModel):
        max_change: float = Field(
            0.1,
            ge=0.02,
            le=0.5,
            title="Max speed change",
            description="Speed varies in 1 ± this; pitch shifts with it.",
        )

    id = "audio_speed"
    label = "Speed change"
    description = "Play the clip faster or slower (pitch shifts with speed)."
    modality = "audio"
    order = 40

    @classmethod
    def apply(cls, sample: AudioClip, params: Params, rng: random.Random, state: Any = None) -> AudioClip:
        import numpy as np

        speed = rng.uniform(1.0 - params.max_change, 1.0 + params.max_change)
        frames = sample.samples.shape[1]
        new_frames = max(1, round(frames / speed))
        positions = np.linspace(0, frames - 1, new_frames)
        source = np.arange(frames)
        resampled = np.stack([np.interp(positions, source, ch) for ch in sample.samples]).astype(np.float32)
        return AudioClip(resampled, sample.sample_rate)
