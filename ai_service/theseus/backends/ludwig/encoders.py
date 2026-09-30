"""Ludwig encoders this backend adds. Currently one: `hf_vision`, any Hugging Face vision backbone.

Ludwig ships encoders for a handful of named families (ResNet, ViT, ConvNeXt, and CLIP/DINOv2/SigLIP each with
its own class) but none that takes an arbitrary `transformers` vision model. This one wraps `AutoModel`, so a
user's own ViT, Swin, DeiT, BEiT, ConvNeXt, EfficientNet, ... (a `config.json` plus `.safetensors` weights) can
be the backbone of an image task without a new class per family.

Safety: `trust_remote_code` is always False, so a model folder cannot make transformers import and run Python
from it; validation (see custom.py) also refuses configs that ask for it. Only safetensors weights are ever
placed on disk (see services/custom_models.py).

This module imports ludwig and torch at module level, so it is imported lazily and MUST be imported before
Ludwig builds or loads a model (train.py and model.py do): Ludwig looks encoders up by name in a registry that
this module fills at import time, both when a config is validated and when a saved model is rebuilt.
"""

import logging

import torch
import torch.nn.functional as F
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.image.pretrained import PretrainedImageEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config

from theseus.backends.ludwig.tasks import HF_VISION

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE_SIZE = 224


@DeveloperAPI
@register_encoder_config(HF_VISION, IMAGE)
class HFVisionEncoderConfig(PretrainedImageEncoderConfig):
    @staticmethod
    def module_name():
        return "HFVisionEncoder"

    type: str = schema_utils.ProtectedString(
        HF_VISION,
        description="Any Hugging Face transformers vision backbone, loaded from a local model folder.",
    )
    pretrained_model_name_or_path: str = schema_utils.String(
        default="",
        description="Local folder of a transformers vision model (config.json and .safetensors weights).",
    )


def _image_size(config) -> tuple[int, int]:
    """The (height, width) the backbone expects, from its config; 224 when it does not say."""
    size = getattr(config, "image_size", None)
    if isinstance(size, int):
        return size, size
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        return int(size[0]), int(size[1])
    if isinstance(size, dict) and "height" in size and "width" in size:
        return int(size["height"]), int(size["width"])
    return _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE


def _pool(outputs) -> torch.Tensor:
    """One feature vector per image, whatever family produced `outputs`."""
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled.flatten(1)  # a conv net's pooler is [B, C, 1, 1]
    hidden = outputs.last_hidden_state
    if hidden.dim() == 3:  # a transformer: [B, tokens, D]
        return hidden.mean(dim=1)
    if hidden.dim() == 4:  # a conv net: [B, C, H, W]
        return hidden.mean(dim=(2, 3))
    raise ValueError(f"Cannot pool a backbone output of shape {tuple(hidden.shape)} into one vector per image")


@DeveloperAPI
@register_encoder(HF_VISION, IMAGE)
class HFVisionEncoder(ImageEncoder):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        use_pretrained: bool = True,
        trainable: bool = True,
        saved_weights_in_checkpoint: bool = False,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = encoder_config

        from transformers import AutoConfig, AutoModel

        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False)
        if use_pretrained and not saved_weights_in_checkpoint:
            logger.info("Loading Hugging Face vision model from %s", pretrained_model_name_or_path)
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False)
        else:
            # A trained model being loaded: its weights come from Ludwig's own checkpoint, so only the
            # architecture (from config.json) is needed here.
            self.model = AutoModel.from_config(hf_config, trust_remote_code=False)

        self._height, self._width = _image_size(hf_config)
        self._channels = int(getattr(hf_config, "num_channels", 3))
        for p in self.model.parameters():
            p.requires_grad_(trainable)
        self._output_dim = self._probe_output_dim()

    def _probe_output_dim(self) -> int:
        """Run one blank image through the backbone to learn its feature size: `hidden_size`, `hidden_sizes`,
        `embed_dim` and friends differ by family, and the output is the only thing that is always right."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                blank = torch.zeros(1, self._channels, self._height, self._width)
                return int(_pool(self.model(pixel_values=blank)).shape[-1])
        finally:
            self.model.train(was_training)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        inputs = inputs.float()
        if inputs.shape[-2:] != (self._height, self._width):
            # Whatever size the pipeline produced, give the backbone the size it was built for.
            inputs = F.interpolate(inputs, size=(self._height, self._width), mode="bilinear", align_corners=False)
        return {ENCODER_OUTPUT: _pool(self.model(pixel_values=inputs))}

    @staticmethod
    def get_schema_cls() -> type[BaseEncoderConfig]:
        return HFVisionEncoderConfig

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_dim])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self._channels, self._height, self._width])
