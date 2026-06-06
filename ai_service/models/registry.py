"""
Model Registry -- a configuration-based registry for Ludwig model variants.

Each variant maps to a Ludwig encoder configuration (using timm models).
No model classes are needed; Ludwig handles all model creation and training.

Usage:
    from ai_service.models.registry import get_model_config, list_models

    config = get_model_config("resnet18")
    # Returns: {"type": "timm", "model_name": "resnet18", "pretrained": True}
"""

from typing import Any

# ──────────────────────────────────────────────────────────────────
# Registry data structure
# ──────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_model(
    variant_id: str,
    *,
    family: str,
    display_name: str,
    description: str = "",
    timm_name: str,
):
    """
    Register a model variant in the global registry.

    Args:
        variant_id: Unique identifier for the variant (e.g., "resnet50").
        family: The model family name (e.g., "resnet").
        display_name: Human-readable name (e.g., "ResNet-50").
        description: Optional description of the variant.
        timm_name: The timm model name used by Ludwig's timm encoder.
    """
    if variant_id in _MODEL_REGISTRY:
        raise ValueError(f"Model variant '{variant_id}' is already registered.")
    _MODEL_REGISTRY[variant_id] = {
        "family": family,
        "display_name": display_name,
        "description": description,
        "timm_name": timm_name,
    }


def get_model_config(variant_id: str) -> dict[str, Any]:
    """Retrieve the Ludwig encoder config for a model variant."""
    if variant_id not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model variant '{variant_id}' not found. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    meta = _MODEL_REGISTRY[variant_id]
    return {
        "type": "timm",
        "model_name": meta["timm_name"],
    }


def get_model_meta(variant_id: str) -> dict[str, Any]:
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


def list_models() -> list[dict[str, Any]]:
    """Return a list of all registered model variants grouped by family."""
    families: dict[str, list[dict]] = {}
    for vid, meta in _MODEL_REGISTRY.items():
        fam = meta["family"]
        if fam not in families:
            families[fam] = []
        families[fam].append(
            {
                "id": vid,
                "display_name": meta["display_name"],
                "description": meta["description"],
            }
        )
    return [
        {"family": fam, "variants": variants}
        for fam, variants in families.items()
    ]
