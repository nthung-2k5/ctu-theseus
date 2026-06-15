"""
Model Registry -- a configuration-based registry for Ludwig model variants.

Each variant maps to a Ludwig encoder configuration (using timm models).
No model classes are needed; Ludwig handles all model creation and training.

Usage:
    from ai_service.models.registry import get_model_config, list_models

    config = get_model_config("resnet18")
    # Returns: {"type": "timm", "model_name": "resnet18", "pretrained": True}
"""

import copy
from dataclasses import dataclass
from typing import Any

# ──────────────────────────────────────────────────────────────────
# Registry data structure
# ──────────────────────────────────────────────────────────────────

@dataclass
class ModelVariant:
    variant_id: str
    display_name: str
    config: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelVariant):
            return self.variant_id == other.variant_id
        elif isinstance(other, str):
            return self.variant_id == other
        else:
            return NotImplemented

@dataclass
class ModelFamily:
    family_id: str
    display_name: str
    variants: list[ModelVariant]


_MODEL_FAMILY: dict[str, ModelFamily] = {}
_MODEL_REGISTRY: dict[str, ModelVariant] = {}

def register_image_model_family_with_variants(
    family_id: str,
    *,
    family_display_name: str,
    variant_key: str,
    variants: list[tuple[str, str]],
):
    """
    Register an image model family in the global registry.

    Args:
        family_id: Unique identifier for the family (e.g., "resnet").
        family_display_name: Human-readable name (e.g., "ResNet").
        variant_key: Key for the variant in the Ludwig encoder configuration.
        variants: List of model variants, each with a unique identifier and display name.
    """

    if family_id in _MODEL_REGISTRY:
        raise ValueError(f"Model family '{family_id}' already registered.")

    _MODEL_FAMILY[family_id] = ModelFamily(
        family_id=family_id,
        display_name=family_display_name,
        variants=list(
            map(
                lambda x: ModelVariant(
                    variant_id=x[0],
                    display_name=x[1],
                    config={
                        "type": family_id,
                        variant_key: x[0],
                    },
                ),
                variants,
            )
        ),
    )

    for variant in _MODEL_FAMILY[family_id].variants:
        _MODEL_REGISTRY[variant.variant_id] = variant

def register_pytorch_image_model_family(
    family_id: str,
    *,
    family_display_name: str,
    variants: list[tuple[str, str]],
):
    """
    Register a PyTorch image model family in the global registry.

    Args:
        family_id: Unique identifier for the family (e.g., "resnet").
        family_display_name: Human-readable name (e.g., "ResNet").
        variants: List of model variants, each with a unique identifier and display name.
    """

    register_image_model_family_with_variants(
        family_id=family_id,
        family_display_name=family_display_name,
        variant_key="model_variant",
        variants=variants,
    )


def get_model_config(variant_id: str) -> dict[str, Any]:
    """Retrieve the Ludwig encoder config for a model variant."""
    if variant_id not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model variant '{variant_id}' not found. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    meta = _MODEL_REGISTRY[variant_id]
    return copy.deepcopy(meta.config)


def get_model_meta(variant_id: str):
    """Retrieve the full metadata for a model variant."""
    if variant_id not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model variant '{variant_id}' not found. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[variant_id]


def has_model(variant_id: str) -> bool:
    """Check if a model variant is registered."""
    return variant_id in _MODEL_REGISTRY


def list_models() -> dict[str, list[tuple[str, str]]]:
    """Return a list of all registered model variants grouped by family."""
    families: dict[str, list[tuple[str, str]]] = {}
    for family in _MODEL_FAMILY.values():
        families[family.display_name] = list(
            map(
                lambda x: (
                    x.variant_id,
                    x.display_name,
                ),
                family.variants,
            )
        )
    return families
