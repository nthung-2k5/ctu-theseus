"""Video preprocessing. Samples are `VideoClip` (uint8 RGB frames, shape frames x height x width x 3, plus fps).

An op that has nothing to do returns the clip it was given, unchanged: `services/preprocessing.py` reads that as
"leave the original in place" and skips the lossy re-encode.
"""

from typing import Any, Literal

from pydantic import Field

from omnitrain.augmentation.media import VideoClip
from omnitrain.preprocessing.base import ParamsModel, Preprocessing


def _even(value: float) -> int:
    """Nearest even size, at least 2: video codecs prefer even dimensions."""
    return max(2, 2 * round(value / 2))


class Resize(Preprocessing):
    class Params(ParamsModel):
        short_edge: int = Field(224, ge=32, le=1080, title="Shorter side (px)")

    id = "video_resize"
    label = "Resize"
    description = (
        "Scale every frame so its shorter side is this many pixels, keeping the aspect ratio. Smaller clips store, "
        "load and train faster."
    )
    modality = "video"
    order = 10

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, state: Any = None) -> VideoClip:
        import cv2
        import numpy as np

        _, height, width = sample.frames.shape[:3]
        scale = params.short_edge / min(height, width)
        new_w, new_h = _even(width * scale), _even(height * scale)
        if (new_w, new_h) == (width, height):
            return sample
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return VideoClip(
            np.stack([cv2.resize(f, (new_w, new_h), interpolation=interpolation) for f in sample.frames]), sample.fps
        )


class ResampleFps(Preprocessing):
    class Params(ParamsModel):
        target_fps: int = Field(15, ge=1, le=60, title="Target frame rate (fps)")

    id = "video_resample_fps"
    label = "Lower frame rate"
    description = "Drop frames to bring every clip down to this frame rate. A clip already at or below it is kept."
    modality = "video"
    order = 20

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, state: Any = None) -> VideoClip:
        import numpy as np

        if sample.fps <= params.target_fps:
            return sample
        count = len(sample.frames)
        new_count = max(1, round(count * params.target_fps / sample.fps))
        picks = np.minimum((np.arange(new_count) * sample.fps / params.target_fps).astype(int), count - 1)
        return VideoClip(sample.frames[picks], float(params.target_fps))


class Trim(Preprocessing):
    class Params(ParamsModel):
        max_seconds: float = Field(
            10.0, ge=1.0, le=300.0, title="Longest clip (seconds)", json_schema_extra={"step": 1}
        )
        keep: Literal["middle", "start"] = Field(
            "middle", title="Keep the", description="Which part of a long clip stays."
        )

    id = "video_trim"
    label = "Trim long clips"
    description = "Cut a clip longer than the limit down to it, from the start or the middle. Shorter clips are kept."
    modality = "video"
    order = 30

    @classmethod
    def apply(cls, sample: VideoClip, params: Params, state: Any = None) -> VideoClip:
        count = len(sample.frames)
        keep = max(1, round(params.max_seconds * sample.fps))
        if count <= keep:
            return sample
        start = 0 if params.keep == "start" else (count - keep) // 2
        return VideoClip(sample.frames[start : start + keep], sample.fps)
