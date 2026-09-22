"""Image augmentations. Samples are PIL images; PIL and numpy are imported inside `apply`."""

import math
import random
from typing import Any

from pydantic import Field

from theseus.augmentation.base import Augmentation, NoParams, ParamsModel


class HorizontalFlip(Augmentation):
    id = "image_horizontal_flip"
    label = "Horizontal flip"
    description = "Mirror the image left to right."
    modality = "vision"
    order = 10

    @classmethod
    def apply(cls, sample: Any, params: NoParams, rng: random.Random, state: Any = None) -> Any:
        from PIL import ImageOps

        return ImageOps.mirror(sample)


class VerticalFlip(Augmentation):
    id = "image_vertical_flip"
    label = "Vertical flip"
    description = "Mirror the image top to bottom."
    modality = "vision"
    order = 20

    @classmethod
    def apply(cls, sample: Any, params: NoParams, rng: random.Random, state: Any = None) -> Any:
        from PIL import ImageOps

        return ImageOps.flip(sample)


class Rotate(Augmentation):
    class Params(ParamsModel):
        max_degrees: float = Field(15.0, ge=1.0, le=45.0, title="Max rotation (degrees)", json_schema_extra={"step": 1})

    id = "image_rotate"
    label = "Rotate"
    description = "Rotate by a random angle up to the maximum, either direction. Corners fill with black."
    modality = "vision"
    order = 30

    @classmethod
    def apply(cls, sample: Any, params: Params, rng: random.Random, state: Any = None) -> Any:
        from PIL import Image

        angle = rng.uniform(-params.max_degrees, params.max_degrees)
        return sample.rotate(angle, resample=Image.Resampling.BICUBIC)


class _Enhance(Augmentation):
    """Shared shape of brightness, contrast and saturation: a random factor around 1.0."""

    class Params(ParamsModel):
        strength: float = Field(0.3, ge=0.05, le=0.9, title="Strength", description="Factor varies in 1 ± strength.")

    enhancer: str

    @classmethod
    def apply(cls, sample: Any, params: Params, rng: random.Random, state: Any = None) -> Any:
        from PIL import ImageEnhance

        factor = rng.uniform(1.0 - params.strength, 1.0 + params.strength)
        return getattr(ImageEnhance, cls.enhancer)(sample).enhance(factor)


class Brightness(_Enhance):
    id = "image_brightness"
    label = "Brightness"
    description = "Make the image randomly brighter or darker."
    modality = "vision"
    order = 40
    enhancer = "Brightness"


class Contrast(_Enhance):
    id = "image_contrast"
    label = "Contrast"
    description = "Randomly raise or lower contrast."
    modality = "vision"
    order = 50
    enhancer = "Contrast"


class Saturation(_Enhance):
    id = "image_saturation"
    label = "Saturation"
    description = "Randomly boost or mute colour. Has no effect on grayscale images."
    modality = "vision"
    order = 60
    enhancer = "Color"


class GaussianBlur(Augmentation):
    class Params(ParamsModel):
        max_radius: float = Field(2.0, ge=0.5, le=5.0, title="Max blur radius (px)")

    id = "image_gaussian_blur"
    label = "Gaussian blur"
    description = "Blur with a random radius up to the maximum."
    modality = "vision"
    order = 70

    @classmethod
    def apply(cls, sample: Any, params: Params, rng: random.Random, state: Any = None) -> Any:
        from PIL import ImageFilter

        return sample.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, params.max_radius)))


class RandomCrop(Augmentation):
    class Params(ParamsModel):
        min_scale: float = Field(
            0.8, ge=0.5, le=0.95, title="Min kept area", description="Fraction of the image area kept, at least."
        )

    id = "image_random_crop"
    label = "Random crop and resize"
    description = "Crop a random region and scale it back to the original size."
    modality = "vision"
    order = 80

    @classmethod
    def apply(cls, sample: Any, params: Params, rng: random.Random, state: Any = None) -> Any:
        from PIL import Image

        width, height = sample.size
        side = math.sqrt(rng.uniform(params.min_scale, 1.0))
        crop_w, crop_h = max(1, round(width * side)), max(1, round(height * side))
        left = rng.randint(0, width - crop_w)
        top = rng.randint(0, height - crop_h)
        return sample.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.Resampling.BICUBIC)


class GaussianNoise(Augmentation):
    class Params(ParamsModel):
        sigma: float = Field(10.0, ge=1.0, le=50.0, title="Noise strength", description="Std dev on a 0-255 scale.")

    id = "image_gaussian_noise"
    label = "Gaussian noise"
    description = "Add random per-pixel noise."
    modality = "vision"
    order = 90

    @classmethod
    def apply(cls, sample: Any, params: Params, rng: random.Random, state: Any = None) -> Any:
        import numpy as np
        from PIL import Image

        gen = np.random.default_rng(rng.getrandbits(64))
        arr = np.asarray(sample, dtype=np.float32)
        noisy = np.clip(arr + gen.normal(0.0, params.sigma, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)  # 2-D gives L, HxWx3 gives RGB: the same modes decode_image produces
