"""
Concrete model implementations using timm (PyTorch Image Models).
Each class is automatically registered via the @register_model decorator.

To add a new model, simply create a new class in this file (or a new file)
and apply the decorator. No other changes are needed.
"""

from ai_service.models.base import TrainableModel

from ai_service.models.registry import register_model

# ────────────────────────────────────────────────────────────────
# ResNet family
# ────────────────────────────────────────────────────────────────
@register_model("resnet18", family="resnet", display_name="ResNet-18")
class ResNet18(TrainableModel):
    timm_name = "resnet18"

@register_model("resnet34", family="resnet", display_name="ResNet-34")
class ResNet34(TrainableModel):
    timm_name = "resnet34"

@register_model("resnet50", family="resnet", display_name="ResNet-50")
class ResNet50(TrainableModel):
    timm_name = "resnet50"

@register_model("resnet101", family="resnet", display_name="ResNet-101")
class ResNet101(TrainableModel):
    timm_name = "resnet101"

@register_model("resnet152", family="resnet", display_name="ResNet-152")
class ResNet152(TrainableModel):
    timm_name = "resnet152"


# ────────────────────────────────────────────────────────────────
# Vision Transformer (ViT) family
# ────────────────────────────────────────────────────────────────
@register_model("vit_tiny", family="vit", display_name="ViT-Tiny")
class ViTTiny(TrainableModel):
    timm_name = "vit_tiny_patch16_224"

@register_model("vit_small", family="vit", display_name="ViT-Small")
class ViTSmall(TrainableModel):
    timm_name = "vit_small_patch16_224"

@register_model("vit_base", family="vit", display_name="ViT-Base")
class ViTBase(TrainableModel):
    timm_name = "vit_base_patch16_224"

@register_model("vit_large", family="vit", display_name="ViT-Large")
class ViTLarge(TrainableModel):
    timm_name = "vit_large_patch16_224"


# ────────────────────────────────────────────────────────────────
# ConvNeXt family
# ────────────────────────────────────────────────────────────────
@register_model("convnext_tiny", family="convnext", display_name="ConvNeXt-Tiny")
class ConvNeXtTiny(TrainableModel):
    timm_name = "convnext_tiny"

@register_model("convnext_small", family="convnext", display_name="ConvNeXt-Small")
class ConvNeXtSmall(TrainableModel):
    timm_name = "convnext_small"

@register_model("convnext_base", family="convnext", display_name="ConvNeXt-Base")
class ConvNeXtBase(TrainableModel):
    timm_name = "convnext_base"

@register_model("convnext_large", family="convnext", display_name="ConvNeXt-Large")
class ConvNeXtLarge(TrainableModel):
    timm_name = "convnext_large"


# ────────────────────────────────────────────────────────────────
# MobileNet family
# ────────────────────────────────────────────────────────────────
@register_model("mobilenet_v2", family="mobilenet", display_name="MobileNetV2")
class MobileNetV2(TrainableModel):
    timm_name = "mobilenetv2_100"

@register_model("mobilenet_v3_small", family="mobilenet", display_name="MobileNetV3-Small")
class MobileNetV3Small(TrainableModel):
    timm_name = "mobilenetv3_small_100"

@register_model("mobilenet_v3_large", family="mobilenet", display_name="MobileNetV3-Large")
class MobileNetV3Large(TrainableModel):
    timm_name = "mobilenetv3_large_100"


# ────────────────────────────────────────────────────────────────
# EfficientNet family
# ────────────────────────────────────────────────────────────────
@register_model("efficientnet_b0", family="efficientnet", display_name="EfficientNet-B0")
class EfficientNetB0(TrainableModel):
    timm_name = "efficientnet_b0"

@register_model("efficientnet_b1", family="efficientnet", display_name="EfficientNet-B1")
class EfficientNetB1(TrainableModel):
    timm_name = "efficientnet_b1"

@register_model("efficientnet_b2", family="efficientnet", display_name="EfficientNet-B2")
class EfficientNetB2(TrainableModel):
    timm_name = "efficientnet_b2"

@register_model("efficientnet_b3", family="efficientnet", display_name="EfficientNet-B3")
class EfficientNetB3(TrainableModel):
    timm_name = "efficientnet_b3"

@register_model("efficientnet_b4", family="efficientnet", display_name="EfficientNet-B4")
class EfficientNetB4(TrainableModel):
    timm_name = "efficientnet_b4"
