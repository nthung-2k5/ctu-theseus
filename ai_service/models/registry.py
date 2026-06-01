"""
Model Registry -- a central factory for dynamically registering and
retrieving model implementations.

Usage:
    from ai_service.models.registry import register_model, get_model, list_models

    @register_model("resnet18", family="resnet", display_name="ResNet-18")
    class ResNet18Model(BaseModel):
        ...

    # Retrieve by variant id
    ModelClass = get_model("resnet18")
    instance = ModelClass(num_classes=10)
"""

from typing import Any, Type

from ai_service.models.base import TrainableModel

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}

def register_model(
    variant_id: str,
    *,
    family: str,
    display_name: str,
    description: str = "",
):
    """
    Decorator to register a model class in the global registry.

    Args:
        variant_id: Unique identifier for the variant (e.g., "resnet50").
        family: The model family name (e.g., "resnet").
        display_name: Human-readable name (e.g., "ResNet-50").
        description: Optional description of the variant.
    """

    def decorator(cls: Type[TrainableModel]) -> Type[TrainableModel]:
        if variant_id in _MODEL_REGISTRY:
            raise ValueError(f"Model variant '{variant_id}' is already registered.")
        _MODEL_REGISTRY[variant_id] = {
            "class": cls,
            "family": family,
            "display_name": display_name,
            "description": description,
        }
        return cls

    return decorator


def get_model(variant_id: str) -> Type[TrainableModel]:
    """Retrieve a model class by its variant id."""
    if variant_id not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model variant '{variant_id}' not found. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[variant_id]["class"]


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
