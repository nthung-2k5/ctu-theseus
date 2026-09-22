"""Tabular augmentations. Samples are the item's features_json dict; only numeric columns are touched."""

import math
import random
from typing import Any

from pydantic import Field

from theseus.augmentation.base import Augmentation, ParamsModel

# (mean, std) per numeric column, computed over the originals being augmented.
ColumnStats = dict[str, tuple[float, float]]


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _column_stats(samples: list[dict[str, Any]]) -> ColumnStats:
    columns: dict[str, list[float]] = {}
    for row in samples:
        for key, value in row.items():
            if _is_number(value):
                columns.setdefault(key, []).append(float(value))
    stats: ColumnStats = {}
    for key, values in columns.items():
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        stats[key] = (mean, std)
    return stats


def _like(original: Any, value: float) -> Any:
    """An int column stays integer-valued."""
    return round(value) if isinstance(original, int) else value


class _NumericOp(Augmentation):
    """Shared: per-column statistics from the originals, applied by the subclass to numeric cells."""

    modality = "tabular"

    @classmethod
    def prepare(cls, samples: Any) -> ColumnStats:
        return _column_stats(list(samples))


class GaussianNoise(_NumericOp):
    class Params(ParamsModel):
        noise_std: float = Field(
            0.1, ge=0.01, le=1.0, title="Noise strength", description="Std dev as a fraction of each column's spread."
        )

    id = "tabular_gaussian_noise"
    label = "Gaussian noise"
    description = "Add random noise to every numeric column, scaled to that column's spread."
    order = 10

    @classmethod
    def apply(cls, sample: dict[str, Any], params: Params, rng: random.Random, state: Any = None) -> dict[str, Any]:
        stats: ColumnStats = state or {}
        out = dict(sample)
        for key, value in sample.items():
            std = stats.get(key, (0.0, 0.0))[1]
            if _is_number(value) and std > 0:
                out[key] = _like(value, value + rng.gauss(0.0, params.noise_std * std))
        return out


class ScaleJitter(_NumericOp):
    class Params(ParamsModel):
        max_change: float = Field(
            0.1, ge=0.01, le=0.5, title="Max change", description="Each value scales by 1 ± this."
        )

    id = "tabular_scale_jitter"
    label = "Scale jitter"
    description = "Multiply every numeric value by a small random factor."
    order = 20

    @classmethod
    def apply(cls, sample: dict[str, Any], params: Params, rng: random.Random, state: Any = None) -> dict[str, Any]:
        out = dict(sample)
        for key, value in sample.items():
            if _is_number(value):
                out[key] = _like(value, value * rng.uniform(1.0 - params.max_change, 1.0 + params.max_change))
        return out


class FeatureDropout(_NumericOp):
    class Params(ParamsModel):
        dropout_rate: float = Field(0.1, ge=0.01, le=0.5, title="Dropout rate", description="Share of cells replaced.")

    id = "tabular_feature_dropout"
    label = "Feature dropout"
    description = "Replace a random share of numeric cells with their column mean, so no one column is relied on."
    order = 30

    @classmethod
    def apply(cls, sample: dict[str, Any], params: Params, rng: random.Random, state: Any = None) -> dict[str, Any]:
        stats: ColumnStats = state or {}
        out = dict(sample)
        for key, value in sample.items():
            if _is_number(value) and key in stats and rng.random() < params.dropout_rate:
                out[key] = _like(value, stats[key][0])
        return out
