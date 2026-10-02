"""Image preprocessing. Samples are PIL images; PIL is imported inside `apply`."""

from typing import Any, Literal

from pydantic import Field

from theseus.preprocessing.base import NoParams, ParamsModel, Preprocessing


class Resize(Preprocessing):
    class Params(ParamsModel):
        width: int = Field(224, ge=8, le=4096, title="Width (px)")
        height: int = Field(224, ge=8, le=4096, title="Height (px)")
        mode: Literal["stretch", "fit"] = Field(
            "stretch",
            title="Mode",
            description="Stretch fills the target size exactly; fit keeps the aspect ratio and pads with black.",
        )

    id = "image_resize"
    label = "Resize"
    description = "Resize every image to a fixed width and height."
    modality = "vision"
    order = 10

    @classmethod
    def apply(cls, sample: Any, params: Params, state: Any = None) -> Any:
        from PIL import Image

        if params.mode == "stretch":
            return sample.resize((params.width, params.height), Image.Resampling.BICUBIC)

        src_w, src_h = sample.size
        scale = min(params.width / src_w, params.height / src_h)
        new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        resized = sample.resize((new_w, new_h), Image.Resampling.BICUBIC)
        canvas = Image.new(sample.mode, (params.width, params.height))
        canvas.paste(resized, ((params.width - new_w) // 2, (params.height - new_h) // 2))
        return canvas


class Grayscale(Preprocessing):
    id = "image_grayscale"
    label = "Grayscale"
    description = "Convert every image to single-channel grayscale."
    modality = "vision"
    order = 20

    @classmethod
    def apply(cls, sample: Any, params: NoParams, state: Any = None) -> Any:
        return sample.copy() if sample.mode == "L" else sample.convert("L")
