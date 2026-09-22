"""Framework-neutral inference response shapes.

Every backend's `LoadedModel.to_output` must produce one of these. The shaping logic itself (how
to read a `category`/`number`/`sequence` prediction column, say) is necessarily framework-specific
and lives with each backend (see `theseus/backends/ludwig/model.py` for Ludwig's).
"""

from typing import Literal, TypedDict

import pandas as pd


class ClassificationClass(TypedDict):
    label: str
    confidence: float


class ClassificationOutput(TypedDict):
    kind: Literal["classification"]
    feature: str
    classes: list[ClassificationClass]


class RegressionOutput(TypedDict):
    kind: Literal["regression"]
    feature: str
    value: float


class TextOutput(TypedDict):
    kind: Literal["text"]
    feature: str
    text: str


class TokenTag(TypedDict):
    token: str
    tag: str


class TokensOutput(TypedDict):
    kind: Literal["tokens"]
    feature: str
    tokens: list[TokenTag]


InferenceOutput = ClassificationOutput | RegressionOutput | TextOutput | TokensOutput


def build_batch_result_frame(input_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Concatenate a batch inference job's original input rows with the backend's own postprocessed
    prediction columns (Ludwig: `{feature}_predictions`, `{feature}_probability`, ...) — the same
    "input + raw prediction columns" shape as the evaluation predictions.parquet (see
    services/evaluate.py), so a batch result and an evaluation export read the same way. Both frames
    come from the same `LoadedModel.predict` call and are therefore already row-aligned; the index
    reset just avoids a spurious join on a non-default input index.
    """
    return pd.concat([input_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
