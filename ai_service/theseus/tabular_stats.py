"""Shared per-column numeric statistics over tabular samples (an item's `features_json` dict).

Used by both the augmentation ops (`augmentation/ops/tabular.py`) and the preprocessing ops
(`preprocessing/ops/tabular.py`), which fit these from different item sets — every original being
augmented vs. the snapshot's train split only — but need the exact same notion of "is this cell
numeric" and the same statistics.
"""

import math
from typing import Any

ColumnStats = dict[str, tuple[float, float]]  # column -> (mean, std)
ColumnBounds = dict[str, tuple[float, float]]  # column -> (min, max)


def is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def like(original: Any, value: float) -> Any:
    """An int column stays integer-valued."""
    return round(value) if isinstance(original, int) else value


def _numeric_columns(samples: list[dict[str, Any]]) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    for row in samples:
        for key, value in row.items():
            if is_number(value):
                columns.setdefault(key, []).append(float(value))
    return columns


def column_stats(samples: list[dict[str, Any]]) -> ColumnStats:
    stats: ColumnStats = {}
    for key, values in _numeric_columns(samples).items():
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        stats[key] = (mean, std)
    return stats


def column_bounds(samples: list[dict[str, Any]]) -> ColumnBounds:
    return {key: (min(values), max(values)) for key, values in _numeric_columns(samples).items()}
