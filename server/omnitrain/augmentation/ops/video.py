"""Video augmentations. Samples are `VideoClip` (uint8 RGB frames, shape frames x height x width x 3, plus fps).

Each keeps the label true: nothing here changes what the clip shows beyond how it is framed, lit or timed. Note that
a flip is wrong for a task where direction is the label (turning left versus right); leave it out there. An op that
cannot change the clip returns it as it was, and `services/augmentation.py` then drops that copy.
"""

import random
from typing import Any

from pydantic import Field

from omnitrain.augmentation.base import Augmentation, NoParams, ParamsModel
from omnitrain.augmentation.media import VideoClip


class HorizontalFlip(Augmentation):
    id = "video_horizontal_flip"
    label = "Horizontal flip"
    description = "Mirror every frame left to right."
    modality = "video"
    order = 10

    @classmethod
    def apply(cls, sample: VideoClip, params: NoParams, rng: random.Random, state: Any = None) -> VideoClip:
        import numpy as np

        return VideoClip(np.ascontiguousarray(sample.frames[:, :, ::-1]), sample.fps)


class TemporalCrop(Augmentation):
    class Params(ParamsModel):
        min_fraction: float = Field(
            0.6, ge=0.3, le=0.95, title="Shortest kept part",
            description="A random stretch of the clip at least this fraction of its length is kept.",
            json_schema_extra={"step": 0.05},
        )  # fmt: skip

    id = "video_temporal_crop"
    label = "Random segment"
    description = "Keep only a random stretch of the clip, so the model cannot rely on where the action starts."
    modality = "video"
    order = 20

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, rng: random.Random, state: Any = None) -> VideoClip:
        count = len(sample.frames)
        keep = max(1, round(count * rng.uniform(params.min_fraction, 1.0)))
        if keep >= count:
            return sample
        start = rng.randint(0, count - keep)
        return VideoClip(sample.frames[start : start + keep], sample.fps)


class Brightness(Augmentation):
    class Params(ParamsModel):
        max_change: float = Field(
            0.25, ge=0.05, le=0.6, title="Max brightness change",
            description="Brightness varies in 1 ± this, the same for the whole clip.",
            json_schema_extra={"step": 0.05},
        )  # fmt: skip

    id = "video_brightness"
    label = "Brightness"
    description = "Make the whole clip randomly brighter or darker. Bright areas clip, never wrap."
    modality = "video"
    order = 30

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, rng: random.Random, state: Any = None) -> VideoClip:
        import cv2
        import numpy as np

        factor = rng.uniform(1.0 - params.max_change, 1.0 + params.max_change)
        # Per frame, saturating: converting the whole clip to float would need four times its memory.
        return VideoClip(np.stack([cv2.convertScaleAbs(f, alpha=factor) for f in sample.frames]), sample.fps)


class Speed(Augmentation):
    class Params(ParamsModel):
        max_change: float = Field(
            0.25, ge=0.05, le=0.5, title="Max speed change",
            description="Speed varies in 1 ± this; frames are repeated or skipped.",
            json_schema_extra={"step": 0.05},
        )  # fmt: skip

    id = "video_speed"
    label = "Speed change"
    description = "Play the clip faster or slower by skipping or repeating frames."
    modality = "video"
    order = 40

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, rng: random.Random, state: Any = None) -> VideoClip:
        import numpy as np

        count = len(sample.frames)
        speed = rng.uniform(1.0 - params.max_change, 1.0 + params.max_change)
        new_count = max(1, round(count / speed))
        picks = np.linspace(0, count - 1, new_count).round().astype(int)
        if new_count == count and np.array_equal(picks, np.arange(count)):
            return sample
        return VideoClip(sample.frames[picks], sample.fps)


class Zoom(Augmentation):
    class Params(ParamsModel):
        min_scale: float = Field(
            0.75, ge=0.5, le=0.95, title="Smallest visible area",
            description="A random window at least this fraction of each side is kept and scaled back up.",
            json_schema_extra={"step": 0.05},
        )  # fmt: skip

    id = "video_zoom"
    label = "Random zoom"
    description = "Zoom into a random part of every frame (the same part for the whole clip)."
    modality = "video"
    order = 50

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, rng: random.Random, state: Any = None) -> VideoClip:
        import cv2
        import numpy as np

        _, height, width = sample.frames.shape[:3]
        scale = rng.uniform(params.min_scale, 1.0)
        win_h, win_w = max(2, round(height * scale)), max(2, round(width * scale))
        if win_h >= height and win_w >= width:
            return sample
        top = round(rng.random() * (height - win_h))
        left = round(rng.random() * (width - win_w))
        window = sample.frames[:, top : top + win_h, left : left + win_w]
        return VideoClip(
            np.stack([cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR) for f in window]), sample.fps
        )
