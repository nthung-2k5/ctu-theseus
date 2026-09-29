"""Tabular preprocessing. Samples are the item's features_json dict; only numeric columns are touched.

Statistics are fit once, from the snapshot's TRAIN split only (see services/preprocessing.py), so
nothing about validation or test leaks into how a value is scaled.
"""

from collections.abc import Sequence
from typing import Any

from theseus.preprocessing.base import NoParams, Preprocessing
from theseus.tabular_stats import ColumnBounds, ColumnStats, column_bounds, column_stats, is_number


class Standardize(Preprocessing):
    """Per-column statistics fit on the train split, applied by the subclass to numeric cells."""

    modality = "tabular"
    id = "tabular_standardize"
    label = "Standardize (z-score)"
    description = "Scale every numeric column to zero mean and unit variance, fit on the train split."
    order = 10

    @classmethod
    def fit(cls, samples: Sequence[Any]) -> ColumnStats:
        return column_stats(list(samples))

    @classmethod
    def apply(cls, sample: dict[str, Any], params: NoParams, state: Any = None) -> dict[str, Any]:
        stats: ColumnStats = state or {}
        out = dict(sample)
        for key, value in sample.items():
            if is_number(value) and key in stats:
                mean, std = stats[key]
                if std > 0:
                    out[key] = (value - mean) / std
        return out


class MinMax(Preprocessing):
    modality = "tabular"
    id = "tabular_min_max"
    label = "Min-max scale"
    description = "Scale every numeric column to the 0-1 range, fit on the train split."
    order = 20

    @classmethod
    def fit(cls, samples: Sequence[Any]) -> ColumnBounds:
        return column_bounds(list(samples))

    @classmethod
    def apply(cls, sample: dict[str, Any], params: NoParams, state: Any = None) -> dict[str, Any]:
        bounds: ColumnBounds = state or {}
        out = dict(sample)
        for key, value in sample.items():
            if is_number(value) and key in bounds:
                lo, hi = bounds[key]
                if hi > lo:
                    out[key] = (value - lo) / (hi - lo)
        return out
