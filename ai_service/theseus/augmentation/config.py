"""Wire models for augmentation: what a snapshot is built with, and what the UI can offer.

Kept free of any heavy import (no PIL, numpy or the registry) so schemas can import it.
"""

from typing import Any, Literal

from pydantic import Field

from theseus.db.enums import DatasetModality
from theseus.schemas.common import ApiModel

# Hard limits on one snapshot's augmentation request.
MAX_COPIES_PER_ITEM = 10
MAX_OPS = 20
# Total augmented items one snapshot may add: bounds S3 writes and build time.
MAX_AUGMENTED_ITEMS = 20_000


class AugmentationOpConfig(ApiModel):
    id: str
    # Chance that this op runs on a given copy.
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    # Validated against the op's Params model; keys are camelCase (see ParamSpec.name).
    params: dict[str, Any] = Field(default_factory=dict)


class AugmentationConfig(ApiModel):
    """How to augment the TRAIN split of a snapshot: N extra copies per original, each through these ops."""

    copies_per_item: int = Field(default=1, ge=1, le=MAX_COPIES_PER_ITEM)
    ops: list[AugmentationOpConfig] = Field(min_length=1, max_length=MAX_OPS)
    # Filled in by the snapshot build: original items that could not be augmented (e.g. undecodable files).
    skipped_items: int | None = None


class ParamSpec(ApiModel):
    """One tunable parameter of an op, flattened so the web can render a form without knowing the op."""

    name: str
    label: str
    description: str | None = None
    type: Literal["int", "float", "bool", "choice"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None


class AugmentationInfo(ApiModel):
    id: str
    label: str
    description: str
    modality: DatasetModality
    params: list[ParamSpec]
