"""String-literal enums shared by the DB models and the API schemas (ported from server/lib/enums.ts)."""

from typing import Literal

# A task is a plugin (omnitrain/tasks/), not an enum: `projects.task` is free text holding a task id.

DatasetModality = Literal["text", "vision", "audio", "video", "tabular"]
SplitType = Literal["train", "test", "validation"]
ImageFormat = Literal["jpeg", "png"]
AudioCodec = Literal["wav", "mp3", "flac", "ogg"]
VideoContainer = Literal["mp4", "webm"]
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
RunEventKind = Literal["status", "metric"]

TERMINAL_RUN_STATUSES: tuple[str, ...] = ("succeeded", "failed", "canceled")
