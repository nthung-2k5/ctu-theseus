"""
Model variant registrations for Ludwig (using timm encoders).

Each call to register_model() adds a variant to the global registry.
The timm_name maps directly to Ludwig's timm encoder model_name parameter.
"""

from ai_service.models.registry import register_model

# ────────────────────────────────────────────────────────────────
# ResNet family
# ────────────────────────────────────────────────────────────────
register_model("resnet18", family="resnet", display_name="ResNet-18", timm_name="resnet18")
register_model("resnet34", family="resnet", display_name="ResNet-34", timm_name="resnet34")
register_model("resnet50", family="resnet", display_name="ResNet-50", timm_name="resnet50")
register_model("resnet101", family="resnet", display_name="ResNet-101", timm_name="resnet101")
register_model("resnet152", family="resnet", display_name="ResNet-152", timm_name="resnet152")

# ────────────────────────────────────────────────────────────────
# Vision Transformer (ViT) family
# ────────────────────────────────────────────────────────────────
register_model("vit_tiny", family="vit", display_name="ViT-Tiny", timm_name="vit_tiny_patch16_224")
register_model("vit_small", family="vit", display_name="ViT-Small", timm_name="vit_small_patch16_224")
register_model("vit_base", family="vit", display_name="ViT-Base", timm_name="vit_base_patch16_224")
register_model("vit_large", family="vit", display_name="ViT-Large", timm_name="vit_large_patch16_224")

# ────────────────────────────────────────────────────────────────
# ConvNeXt family
# ────────────────────────────────────────────────────────────────
register_model("convnext_tiny", family="convnext", display_name="ConvNeXt-Tiny", timm_name="convnext_tiny")
register_model("convnext_small", family="convnext", display_name="ConvNeXt-Small", timm_name="convnext_small")
register_model("convnext_base", family="convnext", display_name="ConvNeXt-Base", timm_name="convnext_base")
register_model("convnext_large", family="convnext", display_name="ConvNeXt-Large", timm_name="convnext_large")

# ────────────────────────────────────────────────────────────────
# MobileNet family
# ────────────────────────────────────────────────────────────────
register_model("mobilenet_v2", family="mobilenet", display_name="MobileNetV2", timm_name="mobilenetv2_100")
register_model("mobilenet_v3_small", family="mobilenet", display_name="MobileNetV3-Small", timm_name="mobilenetv3_small_100")
register_model("mobilenet_v3_large", family="mobilenet", display_name="MobileNetV3-Large", timm_name="mobilenetv3_large_100")

# ────────────────────────────────────────────────────────────────
# EfficientNet family
# ────────────────────────────────────────────────────────────────
register_model("efficientnet_b0", family="efficientnet", display_name="EfficientNet-B0", timm_name="efficientnet_b0")
register_model("efficientnet_b1", family="efficientnet", display_name="EfficientNet-B1", timm_name="efficientnet_b1")
register_model("efficientnet_b2", family="efficientnet", display_name="EfficientNet-B2", timm_name="efficientnet_b2")
register_model("efficientnet_b3", family="efficientnet", display_name="EfficientNet-B3", timm_name="efficientnet_b3")
register_model("efficientnet_b4", family="efficientnet", display_name="EfficientNet-B4", timm_name="efficientnet_b4")
