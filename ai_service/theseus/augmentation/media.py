"""Decode and encode the file-backed samples: images (PIL) and audio (numpy).

Heavy imports (PIL, numpy, scipy, torchaudio) happen inside the functions so importing the
augmentation package, which create_app() does, stays cheap.
"""

import io
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AudioClip:
    # float32 in [-1, 1], shape (channels, frames)
    samples: Any
    sample_rate: int


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
            return AudioClip(np.ascontiguousarray(arr), int(rate))
        except Exception:
            pass  # e.g. an ADPCM or float24 WAV scipy rejects: let torchaudio have a go

    import torchaudio

    waveform, rate = torchaudio.load(io.BytesIO(data))
    return AudioClip(waveform.numpy().astype(np.float32), int(rate))


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


def file_ext(storage_key: str | None) -> str:
    return os.path.splitext(storage_key or "")[1]
