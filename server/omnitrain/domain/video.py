"""Video files: validate and describe an upload, and cut a poster frame for the grid.

Decoding goes through OpenCV (`cv2.VideoCapture`), which bundles its own FFmpeg. torchcodec would be the
natural choice next to torch but needs FFmpeg shared libraries installed on the host, so it can fail to import
on a machine that otherwise works. `cv2` is imported inside the functions: it is a heavy import and the API must
start without it.

Everything here is synchronous and reads from a local path (OpenCV cannot read from memory): call it from async
code through an executor.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from omnitrain.db.enums import VideoContainer

VIDEO_EXTENSIONS = (".mp4", ".webm")
# A clip is decoded in full only when its header lacks a frame count (common for browser-recorded WebM); past
# this many frames a file is refused instead, which bounds the time a single upload can hold a worker.
MAX_COUNTED_FRAMES = 200_000


class VideoError(ValueError):
    """The file is not a video this server can decode. The message is safe to show the uploader."""


class VideoInfo(NamedTuple):
    width: int
    height: int
    duration_seconds: float
    fps: float
    frame_count: int
    container: VideoContainer | None
    codec: str | None


def container_for(content_type: str | None, ext: str) -> VideoContainer | None:
    if content_type == "video/webm" or ext.lower() == ".webm":
        return "webm"
    if content_type == "video/mp4" or ext.lower() in (".mp4", ".m4v"):
        return "mp4"
    return None


def _fourcc(code: float) -> str | None:
    value = int(code)
    text = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    return text if text.isalnum() else None


def _count_frames(cap: Any) -> int:
    cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES
    count = 0
    while count < MAX_COUNTED_FRAMES and cap.grab():
        count += 1
    return count


def probe_path(path: str, container: VideoContainer | None = None) -> VideoInfo:
    """Open the file, read its properties and prove the first frame decodes. Raises `VideoError` otherwise."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("Could not open this file as a video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        ok, _ = cap.read()
        if not ok or width <= 0 or height <= 0:
            raise VideoError("This video has no decodable frames")
        if not 0 < fps <= 1000:
            raise VideoError("This video has no usable frame rate")
        if frames <= 0:
            frames = _count_frames(cap)
            if frames <= 0:
                raise VideoError("This video has no decodable frames")
    finally:
        cap.release()
    return VideoInfo(width, height, round(frames / fps, 3), round(fps, 3), frames, container, codec)


def probe(data: bytes, ext: str, container: VideoContainer | None = None) -> VideoInfo:
    """`probe_path` for an upload held in memory (written to a temp file first)."""
    suffix = ext if ext.lower() in VIDEO_EXTENSIONS else ".mp4"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"clip{suffix}"
        path.write_bytes(data)
        return probe_path(str(path), container)


def poster_jpeg(path: str, size: int) -> bytes:
    """A JPEG of a frame from the middle of the clip, fitted inside `size`x`size` (never upscaled)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None
        if frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
            ok, frame = cap.read()
            frame = frame if ok else None
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                raise VideoError("This video has no decodable frames")
    finally:
        cap.release()
    height, width = frame.shape[:2]
    scale = min(1.0, size / max(width, height))
    if scale < 1.0:
        frame = cv2.resize(
            frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise VideoError("Could not encode a poster frame")
    return encoded.tobytes()


def is_video_key(storage_url: str | None) -> bool:
    return bool(storage_url) and os.path.splitext(storage_url)[1].lower() in VIDEO_EXTENSIONS  # type: ignore[arg-type,type-var]
