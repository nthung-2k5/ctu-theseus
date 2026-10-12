"""Build a torchvision video classifier with a fresh classification head. Imports torch (lazily imported module)."""

from torch import nn
from torchvision.models import video as tv

# Kinetics-400 weights per architecture. Only the ones in tasks.VIDEO_MODELS.
_WEIGHTS = {
    "r3d_18": tv.R3D_18_Weights.KINETICS400_V1,
    "mc3_18": tv.MC3_18_Weights.KINETICS400_V1,
    "r2plus1d_18": tv.R2Plus1D_18_Weights.KINETICS400_V1,
    "s3d": tv.S3D_Weights.KINETICS400_V1,
    "swin3d_t": tv.Swin3D_T_Weights.KINETICS400_V1,
}


def _replace_head(model: nn.Module, model_id: str, num_classes: int) -> nn.Module:
    """Swap the 400-way Kinetics head for a `num_classes` one and return the new head."""
    if model_id == "s3d":
        head = nn.Conv3d(model.classifier[1].in_channels, num_classes, kernel_size=1)
        model.classifier[1] = head
        return model.classifier
    if model_id == "swin3d_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model.head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.fc


def build_model(model_id: str, num_classes: int, *, pretrained: bool, freeze_backbone: bool = False) -> nn.Module:
    if model_id not in _WEIGHTS:
        raise ValueError(f'"{model_id}" is not a video model')
    model = getattr(tv, model_id)(weights=_WEIGHTS[model_id] if pretrained else None)
    head = _replace_head(model, model_id, num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model
