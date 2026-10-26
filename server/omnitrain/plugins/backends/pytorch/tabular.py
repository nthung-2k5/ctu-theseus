"""Snapshot parquet -> the feature matrix and targets a table model trains on. numpy and pandas only: no torch.

`features` is the one place a frame becomes model input, used by training and by serving alike, so a trained model is
always fed what it was fitted on. A missing, non-numeric or non-finite cell becomes NaN, which the network replaces by
the column's fill value itself (see network.TabularNet), so serving a row with a gap works the way training did.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from omnitrain.plugins.backends.pytorch.spec import TabularSpec, safe_std

SPLITS = ("train", "validation", "test")


def features(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> np.ndarray:
    """`[rows, columns]` float32; every cell that is not a finite number is NaN."""
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    # `np.where`, not an in-place assignment: pandas may hand back a read-only view of its own data.
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32)


@dataclass
class Split:
    x: np.ndarray
    # Class indices (int64) for a classifier, the target in its own units (float32) for a regressor.
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def prepare(
    df: pd.DataFrame, *, kind: str, columns: list[str], label_column: str, class_names: list[str], split_column: str
) -> dict[str, Split]:
    """The labelled rows of each snapshot split ("train", "validation", "test") as features and targets. A row whose
    label is missing, or (for a classifier) is not one of `class_names`, has nothing to learn from and is left out."""
    df = df.reset_index(drop=True)
    if kind == "regress":
        target = pd.to_numeric(df[label_column], errors="coerce")
        usable = np.isfinite(target.to_numpy(dtype=float))
        y_all = target.to_numpy(dtype=np.float32)
    else:
        index = {name: i for i, name in enumerate(class_names)}
        codes = df[label_column].map(lambda v: index.get(str(v)) if not pd.isna(v) else None)
        usable = codes.notna().to_numpy()
        y_all = codes.fillna(-1).to_numpy(dtype=np.int64)

    out: dict[str, Split] = {}
    for name in SPLITS:
        rows = usable & (df[split_column] == name).to_numpy()
        out[name] = Split(x=features(df[rows], columns), y=y_all[rows])
    return out


def fit_spec(columns: list[str], x_train: np.ndarray, y_train: np.ndarray | None) -> TabularSpec:
    """Per-column mean, std and fill from the TRAIN rows only, and (for a regressor, given its `y_train`) the target's
    mean and std. A column that is empty in train fills with 0; one that never varies is left unscaled."""
    # An all-empty column makes numpy warn about a mean of nothing; it is handled by the `count` checks below.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        count = np.sum(~np.isnan(x_train), axis=0)
        mean = np.where(count > 0, np.nanmean(x_train, axis=0, dtype=np.float64), 0.0)
        std = np.where(count > 1, np.nanstd(x_train, axis=0, dtype=np.float64), 1.0)
    target_mean, target_std = 0.0, 1.0
    if y_train is not None and len(y_train):
        target_mean, target_std = float(np.mean(y_train, dtype=np.float64)), safe_std(float(np.std(y_train)))
    return TabularSpec(
        columns=tuple(columns),
        mean=tuple(float(v) for v in mean),
        std=tuple(safe_std(float(v)) for v in std),
        fill=tuple(float(v) for v in mean),
        target_mean=target_mean,
        target_std=target_std,
    )


def balanced_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Each class weighted inversely to its share of the train rows (scikit-learn's "balanced"); a class with no train
    rows gets 0, since nothing is ever scored against it."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    present = int((counts > 0).sum())
    weights = np.zeros(num_classes, dtype=np.float64)
    seen = counts > 0
    weights[seen] = len(y_train) / (present * counts[seen])
    return weights.astype(np.float32)
