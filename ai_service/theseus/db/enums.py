"""String-literal enums shared by the DB models and the API schemas (ported from server/lib/enums.ts)."""

from typing import Literal

# fmt: off
ProjectTask = Literal[
    "text_classification", "token_classification", "text_generation", "question_answering",
    "summarization", "sequence_to_sequence", "text_embedding",
    "image_classification", "object_detection", "image_segmentation", "image_captioning",
    "audio_classification", "automatic_speech_recognition", "audio_segmentation", "audio_captioning",
    "tabular_regression", "tabular_classification", "tabular_clustering", "tabular_anomaly_detection",
]
# fmt: on
DatasetModality = Literal["text", "vision", "audio", "tabular"]
SplitType = Literal["train", "test", "validation"]
ImageFormat = Literal["jpeg", "png"]
AudioCodec = Literal["wav", "mp3", "flac", "ogg"]
AnnotationType = Literal[
    "classification", "bounding_box", "segmentation_mask", "text_sequence", "token_tags", "preference_rank"
]
TrainingStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
DatasetVersionStatus = Literal["draft", "building", "ready", "failed"]
ExportStatus = Literal["pending", "converting", "assembling", "ready", "failed"]
EvaluationSplit = Literal["train", "test", "validation", "full"]
EvaluationStatus = Literal["success", "failed"]
SweepStrategy = Literal["grid", "random"]
SweepStatus = Literal["running", "completed", "canceled"]
# Only what the browser consumes. Export and evaluation state live in their own tables.
RunEventKind = Literal["status", "metric", "log"]

TERMINAL_RUN_STATUSES: tuple[str, ...] = ("succeeded", "failed", "canceled")
