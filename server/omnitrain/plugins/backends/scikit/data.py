"""Snapshot parquet -> the feature matrix and labels scikit-learn fits on (pandas only, no scikit-learn import).

`features` is the one place a frame becomes model input, used by training and by serving alike, so a trained model
is always fed what it was fitted on.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


def features(frame: pd.DataFrame, input_kind: str, columns: list[str]) -> Any:
    """Text: one string per row (a missing text is the empty string). Records: a numeric frame, a missing or
    non-numeric cell becoming NaN, which the models either handle or have imputed."""
    if input_kind == "text":
        return frame[columns[0]].fillna("").astype(str)
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def labels(frame: pd.DataFrame, kind: str, column: str) -> pd.Series:
    """Class names as strings (classification) or floats (regression); a missing or unparseable label is NaN/None."""
    if kind == "regress":
        return pd.to_numeric(frame[column], errors="coerce")
    return frame[column].map(lambda v: None if pd.isna(v) else str(v))


@dataclass
class Split:
    x: Any
    y: pd.Series

    def __len__(self) -> int:
        return len(self.y)


def prepare(df: pd.DataFrame, config: dict[str, Any], split_column: str) -> dict[str, Split]:
    """The labelled rows of each snapshot split ("train", "validation", "test") as features and labels."""
    df = df.reset_index(drop=True)
    columns: list[str] = config["inputColumns"]
    input_kind = "text" if "text" in config else "record"
    kind: str = config["task"]
    label_column: str = config["labelColumn"]

    y_all = labels(df, kind, label_column)
    usable = y_all.notna()
    if input_kind == "text":
        usable &= df[columns[0]].notna() & df[columns[0]].astype(str).str.strip().ne("")

    out: dict[str, Split] = {}
    for name in ("train", "validation", "test"):
        rows = df[usable & (df[split_column] == name)]
        out[name] = Split(x=features(rows, input_kind, columns), y=y_all[rows.index])
    return out
