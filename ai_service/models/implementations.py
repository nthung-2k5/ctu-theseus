"""
Model variant registrations for Ludwig (using timm encoders).

Each call to register_model() adds a variant to the global registry.
The model_name maps directly to Ludwig's timm encoder model_name parameter.
"""

from ai_service.models.registry import (
    register_image_model_family_with_variants,
    register_pytorch_image_model_family,
)

# ────────────────────────────────────────────────────────────────
# ResNet family (removed (outdated), use more modern alternative instead)
# ────────────────────────────────────────────────────────────────
# register_model("resnet18", family="resnet", display_name="ResNet-18", model_name="resnet18")
# register_model("resnet34", family="resnet", display_name="ResNet-34", model_name="resnet34")
# register_model("resnet50", family="resnet", display_name="ResNet-50", model_name="resnet50")
# register_model("resnet101", family="resnet", display_name="ResNet-101", model_name="resnet101")
# register_model("resnet152", family="resnet", display_name="ResNet-152", model_name="resnet152")

# ────────────────────────────────────────────────────────────────
# ConvNeXt family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "convnext",
    family_display_name="ConvNeXt",
    variants=[
        ("tiny", "ConvNeXt Tiny"),
        ("small", "ConvNeXt Small"),
        ("base", "ConvNeXt Base"),
        ("large", "ConvNeXt Large"),
    ],
)

register_image_model_family_with_variants(
    "convnextv2",
    family_display_name="ConvNeXt V2",
    variant_key="model_name",
    variants=[
        ("convnextv2_atto", "ConvNeXt V2-Atto"),
        ("convnextv2_femto", "ConvNeXt V2-Femto"),
        ("convnextv2_pico", "ConvNeXt V2-Pico"),
        ("convnextv2_nano", "ConvNeXt V2-Nano"),
        ("convnextv2_tiny", "ConvNeXt V2-Tiny"),
        ("convnextv2_base", "ConvNeXt V2-Base"),
        ("convnextv2_large", "ConvNeXt V2-Large"),
        ("convnextv2_huge", "ConvNeXt V2-Huge"),
        ("convnextv2_atto.fcmae_ft_in1k", "ConvNeXt V2-Atto (FCMaE pre-trained)"),
        ("convnextv2_femto.fcmae_ft_in1k", "ConvNeXt V2-Femto (FCMaE pre-trained)"),
        ("convnextv2_pico.fcmae_ft_in1k", "ConvNeXt V2-Pico (FCMaE pre-trained)"),
        ("convnextv2_nano.fcmae_ft_in1k", "ConvNeXt V2-Nano (FCMaE pre-trained)"),
        ("convnextv2_tiny.fcmae_ft_in1k", "ConvNeXt V2-Tiny (FCMaE pre-trained)"),
        ("convnextv2_base.fcmae_ft_in1k", "ConvNeXt V2-Base (FCMaE pre-trained)"),
        ("convnextv2_large.fcmae_ft_in1k", "ConvNeXt V2-Large (FCMaE pre-trained)"),
        ("convnextv2_huge.fcmae_ft_in1k", "ConvNeXt V2-Huge (FCMaE pre-trained)"),
        (
            "convnextv2_base.fcmae_ft_in22k_in1k",
            "ConvNeXt V2-Base (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K)",
        ),
        (
            "convnextv2_large.fcmae_ft_in22k_in1k",
            "ConvNeXt V2-Large (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K)",
        ),
        (
            "convnextv2_huge.fcmae_ft_in22k_in1k",
            "ConvNeXt V2-Huge (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K)",
        ),
        (
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "ConvNeXt V2-Base (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K, 384x384 Input)",
        ),
        (
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
            "ConvNeXt V2-Large (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K, 384x384 Input)",
        ),
        (
            "convnextv2_huge.fcmae_ft_in22k_in1k_384",
            "ConvNeXt V2-Huge (FCMaE pre-trained, ImageNet-22K -> ImageNet-1K, 384x384 Input)",
        ),
    ],
)

# ────────────────────────────────────────────────────────────────
# EfficientNet family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "efficientnet",
    family_display_name="EfficientNet",
    variants=[
        ("b0", "EfficientNet-B0"),
        ("b1", "EfficientNet-B1"),
        ("b2", "EfficientNet-B2"),
        ("b3", "EfficientNet-B3"),
        ("b4", "EfficientNet-B4"),
        ("b5", "EfficientNet-B5"),
        ("b6", "EfficientNet-B6"),
        ("b7", "EfficientNet-B7"),
        ("v2_s", "EfficientNet-V2-Small"),
        ("v2_m", "EfficientNet-V2-Medium"),
        ("v2_l", "EfficientNet-V2-Large"),
    ],
)

# ────────────────────────────────────────────────────────────────
# MaxViT family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "maxvit",
    family_display_name="MaxViT",
    variants=[
        ("t", "MaxViT"),
    ],
)

# ────────────────────────────────────────────────────────────────
# MNASNet family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "mnasnet",
    family_display_name="MNASNet",
    variants=[
        ("0_5", "MNASNet 0.5x"),
        ("0_75", "MNASNet 0.75x"),
        ("1_0", "MNASNet 1.0x"),
        ("1_3", "MNASNet 1.3x"),
    ],
)

# ────────────────────────────────────────────────────────────────
# MobileNet family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "mobilenetv2",
    family_display_name="MobileNetV2",
    variants=[
        ("base", "MobileNetV2"),
    ],
)

register_pytorch_image_model_family(
    "mobilenetv3",
    family_display_name="MobileNetV3",
    variants=[
        ("small", "MobileNetV3 Small"),
        ("large", "MobileNetV3 Large"),
    ],
)

# ────────────────────────────────────────────────────────────────
# RegNet family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "regnet",
    family_display_name="RegNet",
    variants=[
        ("x_1_6gf", "RegNetX 1.6 GFLOPS"),
        ("x_16gf", "RegNetX 16 GFLOPS"),
        ("x_32gf", "RegNetX 32 GFLOPS"),
        ("x_3_2gf", "RegNetX 3.2 GFLOPS"),
        ("x_400mf", "RegNetX 400 MFLOPS"),
        ("x_800mf", "RegNetX 800 MFLOPS"),
        ("x_8gf", "RegNetX 8 GFLOPS"),
        ("y_128gf", "RegNetY 128 GFLOPS"),
        ("y_16gf", "RegNetY 16 GFLOPS"),
        ("y_1_6gf", "RegNetY 1.6 GFLOPS"),
        ("y_32gf", "RegNetY 32 GFLOPS"),
        ("y_3_2gf", "RegNetY 3.2 GFLOPS"),
        ("y_400mf", "RegNetY 400 MFLOPS"),
        ("y_800mf", "RegNetY 800 MFLOPS"),
        ("y_8gf", "RegNetY 8 GFLOPS"),
    ],
)

# ────────────────────────────────────────────────────────────────
# ShuffleNetV2 family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "shufflenet_v2",
    family_display_name="ShuffleNetV2",
    variants=[
        ("x0_5", "ShuffleNetV2 0.5x"),
        ("x1_0", "ShuffleNetV2 1.0x"),
        ("x1_5", "ShuffleNetV2 1.5x"),
        ("x2_0", "ShuffleNetV2 2.0x"),
    ],
)

# ────────────────────────────────────────────────────────────────
# SqueezeNet family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "squeezenet",
    family_display_name="SqueezeNet",
    variants=[
        ("1_0", "SqueezeNet 1.0"),
        ("1_1", "SqueezeNet 1.1"),
    ],
)

# ────────────────────────────────────────────────────────────────
# SwinTransformer family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "swin_transformer",
    family_display_name="SwinTransformer",
    variants=[
        ("t", "SwinTransformer Tiny"),
        ("s", "SwinTransformer Small"),
        ("b", "SwinTransformer Base"),
    ],
)

# ────────────────────────────────────────────────────────────────
# Vision Transformer (ViT) family
# ────────────────────────────────────────────────────────────────

register_pytorch_image_model_family(
    "vit",
    family_display_name="Vision Transformer (ViT)",
    variants=[
        ("b_16", "ViT Base/16"),
        ("b_32", "ViT Base/32"),
        ("l_16", "ViT Large/16"),
        ("l_32", "ViT Large/32"),
        ("h_14", "ViT Huge/14"),
    ],
)

# ────────────────────────────────────────────────────────────────
# CAFormer family
# ────────────────────────────────────────────────────────────────

register_image_model_family_with_variants(
    "caformer",
    family_display_name="CAFormer",
    variant_key="model_name",
    variants=[
        ("caformer_s18", "CAFormer Small (18 layers)"),
        ("caformer_s36", "CAFormer Small (36 layers)"),
        ("caformer_s44", "CAFormer Large (44 layers)"),
        ("caformer_s64", "CAFormer Extra-Large (64 layers)"),
        (
            "caformer_s18.sail_in22k_ft_in1k",
            "CAFormer Small (18 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "caformer_s18.sail_in22k_ft_in1k_384",
            "CAFormer Small (18 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "caformer_s36.sail_in22k_ft_in1k",
            "CAFormer Small (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "caformer_s36.sail_in22k_ft_in1k_384",
            "CAFormer Small (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "caformer_m36.sail_in22k_ft_in1k",
            "CAFormer Medium (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "caformer_m36.sail_in22k_ft_in1k_384",
            "CAFormer Medium (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "caformer_b36.sail_in22k_ft_in1k",
            "CAFormer Base (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "caformer_b36.sail_in22k_ft_in1k_384",
            "CAFormer Base (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
    ],
)

# ────────────────────────────────────────────────────────────────
# ConvFormer family
# ────────────────────────────────────────────────────────────────

register_image_model_family_with_variants(
    "convformer",
    family_display_name="ConvFormer",
    variant_key="model_name",
    variants=[
        ("convformer_s18", "ConvFormer Small (18 layers)"),
        ("convformer_s36", "ConvFormer Small (36 layers)"),
        ("convformer_m36", "ConvFormer Medium (36 layers)"),
        ("convformer_b36", "ConvFormer Base (36 layers)"),
        (
            "convformer_s18.sail_in22k_ft_in1k",
            "ConvFormer Small (18 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "convformer_s18.sail_in22k_ft_in1k_384",
            "ConvFormer Small (18 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "convformer_s36.sail_in22k_ft_in1k",
            "ConvFormer Small (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "convformer_s36.sail_in22k_ft_in1k_384",
            "ConvFormer Small (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "convformer_m36.sail_in22k_ft_in1k",
            "ConvFormer Medium (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "convformer_m36.sail_in22k_ft_in1k_384",
            "ConvFormer Medium (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
        (
            "convformer_b36.sail_in22k_ft_in1k",
            "ConvFormer Base (36 layers, ImageNet-22K -> ImageNet-1K pre-trained)",
        ),
        (
            "convformer_b36.sail_in22k_ft_in1k_384",
            "ConvFormer Base (36 layers, ImageNet-22K -> ImageNet-1K pre-trained, 384 resolution)",
        ),
    ],
)

# ────────────────────────────────────────────────────────────────
# PoolFormer family
# ────────────────────────────────────────────────────────────────

register_image_model_family_with_variants(
    "poolformer",
    family_display_name="PoolFormer",
    variant_key="model_name",
    variants=[
        ("poolformer_s12", "PoolFormer Small (12 layers)"),
        ("poolformer_s24", "PoolFormer Small (24 layers)"),
        ("poolformer_s36", "PoolFormer Small (36 layers)"),
        ("poolformer_m36", "PoolFormer Medium (36 layers)"),
        ("poolformer_m48", "PoolFormer Medium (48 layers)"),
        ("poolformerv2_s12", "PoolFormer V2 Small (12 layers)"),
        ("poolformerv2_s24", "PoolFormer V2 Small (24 layers)"),
        ("poolformerv2_s36", "PoolFormer V2 Small (36 layers)"),
        ("poolformerv2_m36", "PoolFormer V2 Medium (36 layers)"),
        ("poolformerv2_m48", "PoolFormer V2 Medium (48 layers)"),
    ],
)
