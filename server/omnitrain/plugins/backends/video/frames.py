"""Turn a video file into the fixed-size tensors the network takes: pick frames, decode them, resize, normalise.

The same arithmetic runs at train time, in evaluation and when serving, and `manifest.py` describes it to a client
that implements it by hand, so it is all in this one module. numpy and OpenCV only; torch is imported inside
`to_tensor` so the module stays cheap to import.

A clip is `num_frames` frames taken `stride` frames apart (so it spans `(num_frames - 1) * stride + 1` source
frames). A video shorter than that repeats its last frame rather than failing. Training starts the clip at a random
offset; serving and evaluation take `clips` clips whose starts are spread evenly, so the whole video is covered.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClipSpec:
    num_frames: int
    stride: int
    # Side of the square the network sees.
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def span(self) -> int:
        return (self.num_frames - 1) * self.stride + 1


def clip_starts(frame_count: int, spec: ClipSpec, clips: int, rng: np.random.Generator | None = None) -> list[int]:
    """Where each clip begins. `rng` set: one random start (training). Otherwise `clips` evenly spread starts."""
    room = max(frame_count - spec.span, 0)
    if rng is not None:
        return [int(rng.integers(0, room + 1))]
    if clips <= 1 or room == 0:
        return [room // 2] if clips <= 1 else [0] * clips
    return [round(room * i / (clips - 1)) for i in range(clips)]


def clip_indices(start: int, frame_count: int, spec: ClipSpec) -> list[int]:
    """The `num_frames` frame numbers of a clip, clamped to the last frame for a short video."""
    last = max(frame_count - 1, 0)
    return [min(start + i * spec.stride, last) for i in range(spec.num_frames)]


def read_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode the frames numbered in `wanted` (RGB, uint8, HxWx3), going forward through the file once.

    Sequential `grab()` is used instead of seeking: a seek in an MP4 lands on the nearest keyframe and is
    unreliable frame-for-frame, and decoding every frame up to the last one wanted costs no more than a seek would
    on the clip lengths this handles. Frames the container cannot deliver are simply absent from the result.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    found: dict[int, np.ndarray] = {}
    try:
        if not cap.isOpened():
            raise ValueError("Could not open the video")
        last = max(wanted)
        for number in range(last + 1):
            if not cap.grab():
                break
            if number in wanted:
                ok, bgr = cap.retrieve()
                if ok:
                    found[number] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return found


def frame_count_of(path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _resize_short_edge(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = size / min(height, width)
    new_w, new_h = max(size, round(width * scale)), max(size, round(height * scale))
    if (new_w, new_h) == (width, height):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def crop_clip(frames: list[np.ndarray], size: int, rng: np.random.Generator | None, flip: bool = False) -> np.ndarray:
    """Resize each frame's SHORTER edge to `size`, then crop a `size` square: centred, or at one random place for the
    whole clip when `rng` is given (training). `flip` mirrors the clip left-right. Returns T x size x size x 3."""
    resized = [_resize_short_edge(f, size) for f in frames]
    height, width = resized[0].shape[:2]
    if rng is None:
        top, left = (height - size) // 2, (width - size) // 2
    else:
        top, left = int(rng.integers(0, height - size + 1)), int(rng.integers(0, width - size + 1))
    clip = np.stack([f[top : top + size, left : left + size] for f in resized])
    return clip[:, :, ::-1] if flip else clip


def load_clips(
    path: str,
    spec: ClipSpec,
    clips: int = 1,
    *,
    rng: np.random.Generator | None = None,
    flip: bool = False,
    frame_count: int | None = None,
) -> list[np.ndarray]:
    """`clips` arrays of T x size x size x 3 uint8 (one random clip when `rng` is given). Raises `ValueError` if the
    file has no decodable frames."""
    count = frame_count if frame_count else frame_count_of(path)
    if count <= 0:
        raise ValueError("The video has no frames")
    starts = clip_starts(count, spec, clips, rng)
    plans = [clip_indices(s, count, spec) for s in starts]
    decoded = read_frames(path, {n for plan in plans for n in plan})
    if not decoded:
        raise ValueError("The video has no decodable frames")
    out = []
    for plan in plans:
        # A frame the decoder failed to deliver (a header that over-states the count) is replaced by the nearest
        # earlier one that did arrive, or the first, so a slightly wrong frame count never fails a whole clip.
        frames, previous = [], decoded[min(decoded)]
        for n in plan:
            previous = decoded.get(n, previous)
            frames.append(previous)
        out.append(crop_clip(frames, spec.size, rng, flip))
    return out


def to_tensor(clip: Any, spec: ClipSpec) -> Any:
    """T x H x W x 3 uint8 -> 3 x T x H x W float32, scaled to 0..1 then normalised with the model's mean/std."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(clip)).permute(3, 0, 1, 2).float() / 255.0
    mean = torch.tensor(spec.mean).view(3, 1, 1, 1)
    std = torch.tensor(spec.std).view(3, 1, 1, 1)
    return (tensor - mean) / std
