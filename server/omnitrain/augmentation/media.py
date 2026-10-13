"""Decode and encode the file-backed samples: images (PIL), audio (numpy) and video (OpenCV, numpy).

Heavy imports (PIL, numpy, scipy, torchaudio, cv2) happen inside the functions so importing the
augmentation package, which create_app() does, stays cheap.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AudioClip:
    # float32 in [-1, 1], shape (channels, frames)
    samples: Any
    sample_rate: int


@dataclass
class VideoClip:
    # uint8 RGB, shape (frames, height, width, 3)
    frames: Any
    fps: float


@dataclass
class Encoded:
    data: bytes
    ext: str
    content_type: str
    # Modality feature columns for the new item's features row.
    features: dict[str, Any]


# -- Images ----------------------------------------------------------------------------------


def decode_image(data: bytes) -> Any:
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img)
    # Ops (enhance, noise) are defined for grayscale and RGB; palette and alpha modes are flattened.
    return img if img.mode in ("RGB", "L") else img.convert("RGB")


def encode_image(img: Any, source_ext: str) -> Encoded:
    """Same container as the original where it is JPEG or PNG; anything else is re-saved as PNG."""
    buf = io.BytesIO()
    if source_ext.lower() in (".jpg", ".jpeg"):
        img.save(buf, format="JPEG", quality=92)
        ext, fmt, ctype = source_ext.lower(), "jpeg", "image/jpeg"
    else:
        img.save(buf, format="PNG")
        ext, fmt, ctype = ".png", "png", "image/png"
    width, height = img.size
    return Encoded(
        buf.getvalue(),
        ext,
        ctype,
        {"width": width, "height": height, "channels": len(img.getbands()), "image_format": fmt},
    )


# -- Audio -----------------------------------------------------------------------------------


def decode_audio(data: bytes, ext: str) -> AudioClip:
    """WAV through scipy (no system codecs needed); other containers through torchaudio."""
    import numpy as np

    if ext.lower() == ".wav":
        try:
            from scipy.io import wavfile

            rate, raw = wavfile.read(io.BytesIO(data))
            arr = np.asarray(raw)
            if arr.dtype == np.uint8:
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            elif np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
            else:
                arr = arr.astype(np.float32)
            arr = arr[None, :] if arr.ndim == 1 else arr.T
            return AudioClip(np.ascontiguousarray(arr), rate)
        except Exception:
            pass  # e.g. an ADPCM or float24 WAV scipy rejects: let torchaudio have a go

    import torchaudio

    waveform, rate = torchaudio.load(io.BytesIO(data))
    return AudioClip(waveform.numpy().astype(np.float32), rate)


def encode_audio(clip: AudioClip) -> Encoded:
    """Always 16-bit PCM WAV: lossless enough for training and decodable everywhere."""
    import numpy as np
    from scipy.io import wavfile

    pcm = (np.clip(clip.samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, clip.sample_rate, pcm.T if pcm.shape[0] > 1 else pcm[0])
    channels, frames = clip.samples.shape
    return Encoded(
        buf.getvalue(),
        ".wav",
        "audio/wav",
        {
            "duration_seconds": round(frames / clip.sample_rate, 3),
            "sample_rate_hz": clip.sample_rate,
            "channels": channels,
            "audio_codec": "wav",
        },
    )


# -- Video -----------------------------------------------------------------------------------

# A video is decoded whole, so a long or large one could take all of a worker's memory. Past this many decoded
# bytes (about 1 GB: 5 minutes of 360p at 30 fps, or 12 seconds of 1080p) it is refused and the item skipped.
MAX_DECODED_VIDEO_BYTES = 1_000_000_000


def decode_video(data: bytes, ext: str) -> VideoClip:
    """Every frame of a clip as RGB. Raises `ValueError` for a file OpenCV cannot read or one too large to hold."""
    import cv2
    import numpy as np

    suffix = ext.lower() if ext.lower() in (".mp4", ".webm") else ".mp4"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"clip{suffix}"
        path.write_bytes(data)
        cap = cv2.VideoCapture(str(path))
        frames: list[Any] = []
        try:
            if not cap.isOpened():
                raise ValueError("Could not open the video")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0 or not 0 < fps <= 1000:
                raise ValueError("The video has no usable size or frame rate")
            frame_bytes = width * height * 3
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if (len(frames) + 1) * frame_bytes > MAX_DECODED_VIDEO_BYTES:
                    raise ValueError("The video is too large to process")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    if not frames:
        raise ValueError("The video has no decodable frames")
    return VideoClip(np.stack(frames), fps)


def encode_video(clip: VideoClip) -> Encoded:
    """Always VP8 in WebM: every browser plays it (MPEG-4 part 2 in MP4, the other writer OpenCV always has, is not
    played by Chrome, and H.264 needs a library the server may not have) and OpenCV reads it back for training.
    Lossy, like any re-encode; the file is read back to prove it decodes before it is stored."""
    import cv2

    from omnitrain.domain import video as video_service

    count, height, width = clip.frames.shape[:3]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.webm"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), clip.fps, (width, height))
        if not writer.isOpened():
            raise ValueError("This server cannot write WebM video")
        try:
            for frame in clip.frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        info = video_service.probe_path(str(path), "webm")
        data = path.read_bytes()
    return Encoded(
        data,
        ".webm",
        "video/webm",
        {
            "width": info.width,
            "height": info.height,
            "duration_seconds": info.duration_seconds,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "container": "webm",
            "codec": info.codec,
        },
    )


# -- By modality -----------------------------------------------------------------------------

FILE_MODALITIES = ("vision", "audio", "video")


def decode_sample(modality: str, data: bytes, ext: str) -> Any:
    """The sample object ops work on, for a file-backed modality."""
    if modality == "vision":
        return decode_image(data)
    if modality == "audio":
        return decode_audio(data, ext)
    if modality == "video":
        return decode_video(data, ext)
    raise ValueError(f"{modality} items are not files")


def encode_sample(modality: str, sample: Any, source_ext: str) -> Encoded:
    if modality == "vision":
        return encode_image(sample, source_ext)
    if modality == "audio":
        return encode_audio(sample)
    if modality == "video":
        return encode_video(sample)
    raise ValueError(f"{modality} items are not files")


def worker_count(modality: str) -> int:
    """Items transformed at once. A decoded clip is hundreds of MB, so video goes one at a time."""
    return 1 if modality == "video" else 4


def file_ext(storage_key: str | None) -> str:
    return os.path.splitext(storage_key or "")[1]
