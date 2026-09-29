"""Wire models for preprocessing: what a snapshot is built with, and what the UI can offer.

Kept free of any heavy import (no PIL, numpy or the registry) so schemas can import it.
"""

from typing import Any, Literal

from pydantic import Field, field_validator

from theseus.db.enums import DatasetModality
from theseus.schemas.common import ApiModel, ParamSpec

__all__ = [
    "MAX_OPS", "MAX_PREPROCESSED_ITEMS", "PreprocessingConfig", "PreprocessingInfo", "ParamSpec",
]  # fmt: skip

MAX_OPS = 20
# Pool items one snapshot may replace with a preprocessed copy: bounds S3 writes and build time.
MAX_PREPROCESSED_ITEMS = 200_000

SplitName = Literal["train", "validation", "test"]
_ALL_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class PreprocessingOpConfig(ApiModel):
    id: str
    # Which splits this op runs on; the preprocessed item replaces the original for exactly these.
    splits: list[SplitName] = Field(default_factory=lambda: list(_ALL_SPLITS), min_length=1)
    # Validated against the op's Params model; keys are camelCase (see ParamSpec.name).
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("splits")
    @classmethod
    def _no_duplicates(cls, value: list[SplitName]) -> list[SplitName]:
        if len(set(value)) != len(value):
            raise ValueError("splits must not repeat")
        return value


class PreprocessingConfig(ApiModel):
    """How to preprocess a snapshot: deterministic ops, each scoped to the splits it replaces."""

    ops: list[PreprocessingOpConfig] = Field(min_length=1, max_length=MAX_OPS)
    # Filled in by the snapshot build: original items that could not be preprocessed (e.g. undecodable files).
    skipped_items: int | None = None


class PreprocessingInfo(ApiModel):
    id: str
    label: str
    description: str
    modality: DatasetModality
    params: list[ParamSpec]
