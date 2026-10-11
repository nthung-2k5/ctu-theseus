"""On-demand JPEG thumbnails for image items, cached in S3 next to the original (see storage.thumbnail_key).

A grid of full-size images is the slowest thing the dataset page does; a thumbnail is a few KB. They are
made on first request rather than at upload, so items uploaded before this existed get them too, and nothing
here can fail an upload. Everything is synchronous: call it from async code through an executor.
"""

import io
import logging

from omnitrain import constants as C
from omnitrain.services import storage

logger = logging.getLogger(__name__)

# The upload path only accepts these (see ItemSpec.accept for the vision tasks).
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# What the item listing links to: big enough for a ~250px card at 2x pixel density.
LIST_SIZE = 256


def is_thumbnailable(storage_url: str | None) -> bool:
    return bool(storage_url) and storage_url.lower().endswith(_IMAGE_EXTENSIONS)  # type: ignore[union-attr]


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
        thumbnail = render_thumbnail(storage.download_bytes(C.BUCKET_DATASETS, storage_url), size)
        storage.upload_bytes(C.BUCKET_DATASETS, key, thumbnail, "image/jpeg")
    return key
