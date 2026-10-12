"""The evaluation report for a video classifier: the shared classification report (confusion matrix, per-class
precision/recall/F1, top errors), scored with `clipsPerVideo` clips per video."""

import pandas as pd

from omnitrain.backends.base import EvalResult
from omnitrain.backends.video.model import VideoLoadedModel
from omnitrain.services.classification_report import evaluate_classification
from omnitrain.services.evaluate import MAX_CLASSES_FOR_MATRIX

# Videos scored per predict call. Clips are scored one video at a time anyway; this just bounds a frame's size.
PREDICT_BATCH = 8


def evaluate(model: VideoLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    return evaluate_classification(
        model, df, split_column, item_id_column,
        label_column=model.meta["outputName"], batch_size=PREDICT_BATCH, max_classes_for_matrix=MAX_CLASSES_FOR_MATRIX,
    )  # fmt: skip
