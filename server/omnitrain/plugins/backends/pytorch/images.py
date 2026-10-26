"""Turn an image file into the tensor a PyTorch classifier takes: the one definition train, serve and evaluate share.

`preprocess` is exactly what the generated export clients do to an image (`imageResize.mode ==
"shorter_edge_center_crop"` in the manifest, see manifest.py): RGB, resize the SHORTER edge to `spec.size` with PIL's
bilinear filter (the other edge truncated to a whole pixel), centre-crop a square, scale to 0..1, then
(x - mean) / std per channel. Doing it with PIL here, rather than a torchvision transform, is what keeps a prediction
made in a client identical to the one made on the server. Training adds random cropping and flipping on top
(`augment`), but a validation or test image always goes through `preprocess`.

Imports PIL and numpy only: no torch.
"""

import numpy as np
from PIL import Image

from omnitrain.plugins.backends.pytorch.spec import ImageSpec

# The range of the visible area `augment` keeps, and of its aspect ratio (torchvision's RandomResizedCrop, with a
# gentler lower bound: a class that only shows in a corner of a picture should not be cropped away too often).
CROP_SCALE = (0.35, 1.0)
CROP_RATIO = (3 / 4, 4 / 3)


def shorter_edge_geometry(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    """(resized_width, resized_height, crop_left, crop_top): scale so the SHORTER edge is `size`, then centre-crop.

    The same arithmetic as torchvision's `Resize(size)` followed by `CenterCrop(size)` and as every generated client's
    `_shorter_edge_geometry`: the longer edge is truncated to a whole pixel, an odd leftover rounds half to even."""
    if width <= height:
        new_width, new_height = size, int(size * height / width)
    else:
        new_width, new_height = int(size * width / height), size
    return new_width, new_height, round((new_width - size) / 2.0), round((new_height - size) / 2.0)


def open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def resize_crop(img: Image.Image, size: int) -> Image.Image:
    new_width, new_height, left, top = shorter_edge_geometry(img.width, img.height, size)
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return img.crop((left, top, left + size, top + size))


def to_array(img: Image.Image, spec: ImageSpec) -> np.ndarray:
    """A `size` x `size` RGB image as normalised float32, channels first: `[3, size, size]`."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def preprocess(path: str, spec: ImageSpec) -> np.ndarray:
    """What serving, evaluation and the export clients do to an image file."""
    return to_array(resize_crop(open_rgb(path), spec.size), spec)


def augment(path: str, spec: ImageSpec, rng: np.random.Generator, *, flip: bool) -> np.ndarray:
    """A random crop of the image (a random part of it, a little stretched), resized to the square the network sees,
    and mirrored left to right half the time when `flip`. Training only."""
    img = open_rgb(path)
    width, height = img.size
    area = width * height
    box = (0, 0, width, height)
    for _ in range(10):
        target = area * rng.uniform(*CROP_SCALE)
        ratio = float(np.exp(rng.uniform(np.log(CROP_RATIO[0]), np.log(CROP_RATIO[1]))))
        crop_width, crop_height = round(np.sqrt(target * ratio)), round(np.sqrt(target / ratio))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left, top = int(rng.integers(0, width - crop_width + 1)), int(rng.integers(0, height - crop_height + 1))
            box = (left, top, left + crop_width, top + crop_height)
            break
    # No suitable crop found in ten tries (a very thin image): keep the whole picture, as torchvision does.
    img = img.resize((spec.size, spec.size), Image.Resampling.BILINEAR, box=box)
    if flip and bool(rng.integers(0, 2)):
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return to_array(img, spec)
