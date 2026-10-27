"""Framework-neutral inference response shapes.

Every backend's `LoadedModel.to_output` must produce one of these. The shaping logic itself (how
to read a `category`/`number`/`sequence` prediction column, say) is necessarily framework-specific
and lives with each backend (see `omnitrain/plugins/backends/pytorch/model.py` for the reference one).
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


class BoundingBoxOut(TypedDict):
    """Normalised to the image (0..1), top-left corner and size: the convention bounding-box annotations use
    (see domain/boxes.py), so a box drawn by a user and a box predicted by a model are interchangeable."""

    x: float
    y: float
    w: float
    h: float


class Detection(TypedDict):
    label: str
    confidence: float
    box: BoundingBoxOut


class DetectionsOutput(TypedDict):
    kind: Literal["detections"]
    feature: str
    # The image's size in pixels, for a client that wants pixel coordinates (x * imageWidth, ...).
    imageWidth: int
    imageHeight: int
    # Most confident first.
    detections: list[Detection]


InferenceOutput = ClassificationOutput | RegressionOutput | TextOutput | TokensOutput | DetectionsOutput


def build_batch_result_frame(input_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Concatenate a batch inference job's original input rows with the backend's own postprocessed
    prediction columns (`{feature}_predictions`, `{feature}_probability`, ...) — the same
    "input + raw prediction columns" shape as the evaluation predictions.parquet (see
    domain/evaluate.py), so a batch result and an evaluation export read the same way. Both frames
    come from the same `LoadedModel.predict` call and are therefore already row-aligned; the index
    reset just avoids a spurious join on a non-default input index.
    """
    return pd.concat([input_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
