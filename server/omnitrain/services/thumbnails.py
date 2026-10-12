"""On-demand JPEG thumbnails for image items (a poster frame for video), cached in S3
next to the original (see storage.thumbnail_key).

A grid of full-size images is the slowest thing the dataset page does; a thumbnail is a few KB. They are
made on first request rather than at upload, so items uploaded before this existed get them too, and nothing
here can fail an upload. Everything is synchronous: call it from async code through an executor.
"""

import io
import logging
import os
import tempfile

from omnitrain import constants as C
from omnitrain.services import storage, video

logger = logging.getLogger(__name__)

# The upload path only accepts these (see ItemSpec.accept for the vision tasks; video has its own list).
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# What the item listing links to: big enough for a ~250px card at 2x pixel density.
LIST_SIZE = 256


def is_thumbnailable(storage_url: str | None) -> bool:
    return bool(storage_url) and storage_url.lower().endswith((*_IMAGE_EXTENSIONS, *video.VIDEO_EXTENSIONS))  # type: ignore[union-attr]


def _render_video_poster(storage_url: str, size: int) -> bytes:
    """A video is fetched to a temp file (OpenCV reads paths, and a clip is too big to hold twice in memory)."""
    with tempfile.TemporaryDirectory() as directory:
        local = storage.download_file(
            C.BUCKET_DATASETS, storage_url, os.path.join(directory, "clip" + os.path.splitext(storage_url)[1])
        )
        return video.poster_jpeg(local, size)


def render_thumbnail(data: bytes, size: int) -> bytes:
    """Fit the image inside `size`x`size` (never upscaling) and encode it as JPEG.

    Raises whatever Pillow raises for an undecodable or oversized (decompression-bomb) image.
    """
    from PIL import Image, ImageOps  # lazy: only the thumbnail path needs it

    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((size, size))
        if image.mode != "RGB":
            image = image.convert("RGB")  # JPEG has no alpha or palette
        out = io.BytesIO()
        image.save(out, "JPEG", quality=82, optimize=True)
        return out.getvalue()


def ensure_thumbnail(storage_url: str, size: int) -> str:
    """The cache key of this object's thumbnail at `size`, generating and storing it if it isn't there yet."""
    key = storage.thumbnail_key(storage_url, size)
    if not storage.file_exists(C.BUCKET_DATASETS, key):
        if video.is_video_key(storage_url):
            thumbnail = _render_video_poster(storage_url, size)
        else:
            thumbnail = render_thumbnail(storage.download_bytes(C.BUCKET_DATASETS, storage_url), size)
        storage.upload_bytes(C.BUCKET_DATASETS, key, thumbnail, "image/jpeg")
    return key
