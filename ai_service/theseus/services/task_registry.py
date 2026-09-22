"""Task registry: the single source of truth for every trainable task, independent of any
training framework.

Ported from server/lib/tasks/{registry,types,index}.ts, then split when trainer backends became a
plugin system (theseus/backends/): what used to live here as a Ludwig-only `ludwig:
LudwigTaskConfig` field (encoder catalogs, Ludwig feature types, trainer knobs) now lives in
`theseus/backends/ludwig/tasks.py`. This module keeps exactly what's true of a task no matter which
backend eventually trains it: what a dataset item for it looks like, its ground-truth shape, the
snapshot columns the builder emits, and the request/response shape inference needs.

It drives project creation, upload validation, snapshot column derivation, inference request/
response shapes and export READMEs on the backend, and (via schema/task_registry.json, generated
by scripts/export_schema.py) the task picker and per-modality UI on the frontend.

Whether a task is actually trainable right now is a runtime question, not a static one: see
`theseus.backends.registry.trainable_backends`. `status` here only distinguishes tasks Theseus can
represent at all ("planned" ones have no annotation/columns story yet, e.g. bounding boxes) from
ones that are just waiting on a backend plugin.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from theseus import constants as C
from theseus.db.enums import AnnotationType, DatasetModality, ProjectTask
from theseus.schemas.common import ApiModel

ColumnKind = Literal[
    "storage_uri", "inline_text", "label", "text_sequence_label", "split", "split_index", "scalar", "item_id"
]
TaskOutputKind = Literal["classification", "regression", "text", "tokens"]
InferenceOutputKind = TaskOutputKind


class ColumnSpec(ApiModel):
    """A column the snapshot builder writes into a version dataset.parquet, in order."""

    name: str
    kind: ColumnKind


class ItemSpec(ApiModel):
    """What one dataset item carries. Drives upload, validation and the pool UI."""

    payload: Literal["file", "inline_text", "record"]
    accept: list[str] | None = None


class AnnotationSpec(ApiModel):
    """Ground truth shape. Drives the Classes page and annotation editors."""

    type: AnnotationType
    requires_label_classes: bool


class TaskOutput(ApiModel):
    """What a trained model for this task predicts. Backend-neutral: every backend that supports
    the task must produce this shape, whatever its own internal feature/output typing looks like."""

    column: str
    kind: TaskOutputKind


class TaskDescriptor(ApiModel):
    id: ProjectTask
    label: str
    modality: DatasetModality
    status: Literal["stable", "experimental", "planned"]
    item_spec: ItemSpec
    annotation: AnnotationSpec
    # Parquet columns the snapshot builder emits, in order.
    columns: list[ColumnSpec]
    # Inline-text input field names, in the order a text-payload task's model expects them. Empty
    # for file payload tasks (exactly one unnamed input) and record payload tasks (tabular: the
    # input columns are dataset-defined, derived from the snapshot instead).
    input_fields: list[str] = []
    # None only for a "planned" task: nothing (this registry or any backend) knows its output shape yet.
    output: TaskOutput | None = None


@dataclass
class SnapshotContext:
    """Everything a backend's `compile` needs to know about one dataset version."""

    columns: list[ColumnSpec] = field(default_factory=list)
    # Label class names in a stable order. Empty for non-classification tasks.
    label_class_names: list[str] = field(default_factory=list)
    # Item count per class NAME (classification only), so a backend can build balanced class
    # weights without a DB round trip. Keyed by name, not index: a backend resolves a name-keyed
    # dict against its own vocabulary, so nothing here needs to predict the index it will assign.
    class_counts: dict[str, int] | None = None


# -- Column shorthands -------------------------------------------------------------------------

IMAGE_PATH = C.IMAGE_PATH_COLUMN_NAME
CLASS = C.CLASS_COLUMN_NAME
SPLIT = C.SPLIT_COLUMN_NAME

_AUDIO_MIME = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"]
_IMAGE_MIME = ["image/jpeg", "image/png"]


def _col(name: str, kind: ColumnKind) -> ColumnSpec:
    return ColumnSpec(name=name, kind=kind)


_CLASSIFY = AnnotationSpec(type="classification", requires_label_classes=True)
_FREE_TEXT = AnnotationSpec(type="text_sequence", requires_label_classes=False)
_INLINE = ItemSpec(payload="inline_text")
_RECORD = ItemSpec(payload="record")


def _task(
    id: ProjectTask,
    label: str,
    modality: DatasetModality,
    status: Literal["stable", "experimental"],
    item_spec: ItemSpec,
    annotation: AnnotationSpec,
    columns: list[ColumnSpec],
    *,
    output: TaskOutput,
    input_fields: list[str] | None = None,
) -> TaskDescriptor:
    return TaskDescriptor(
        id=id, label=label, modality=modality, status=status, item_spec=item_spec, annotation=annotation,
        columns=columns, input_fields=input_fields or [], output=output,
    )  # fmt: skip


def _planned(
    id: ProjectTask, label: str, modality: DatasetModality, ann: AnnotationType, needs_classes: bool
) -> TaskDescriptor:
    """Genuinely blocked, not just unbuilt: Theseus has no bounding-box/mask annotation editor or
    snapshot columns yet, so no backend could support this task even in principle today."""
    return TaskDescriptor(
        id=id,
        label=label,
        modality=modality,
        status="planned",
        item_spec=ItemSpec(payload="file"),
        annotation=AnnotationSpec(type=ann, requires_label_classes=needs_classes),
        columns=[],
    )


def _llm_text_task(id: ProjectTask, label: str, in_cols: list[str], out_col: str) -> TaskDescriptor:
    return _task(
        id, label, "text", "experimental", _INLINE, _FREE_TEXT,
        [*(_col(c, "inline_text") for c in in_cols), _col(out_col, "inline_text"), _col(SPLIT, "split")],
        output=TaskOutput(column=out_col, kind="text"), input_fields=in_cols,
    )  # fmt: skip


def _file_caption_task(
    id: ProjectTask, label: str, modality: DatasetModality, path_col: str, out_col: str, mime: list[str]
) -> TaskDescriptor:
    return _task(
        id, label, modality, "experimental", ItemSpec(payload="file", accept=mime), _FREE_TEXT,
        [_col(path_col, "storage_uri"), _col(out_col, "text_sequence_label"), _col(SPLIT, "split")],
        output=TaskOutput(column=out_col, kind="text"),
    )  # fmt: skip


# -- Tasks ---------------------------------------------------------------------------------------

_TASKS: list[TaskDescriptor] = [
    # -- Tier 1: stable --
    _task(
        "image_classification", "Image Classification", "vision", "stable",
        ItemSpec(payload="file", accept=_IMAGE_MIME), _CLASSIFY,
        [_col(IMAGE_PATH, "storage_uri"), _col(CLASS, "label"), _col(SPLIT, "split")],
        output=TaskOutput(column=CLASS, kind="classification"),
    ),
    _task(
        "text_classification", "Text Classification", "text", "stable", _INLINE, _CLASSIFY,
        [_col("text", "inline_text"), _col(CLASS, "label"), _col(SPLIT, "split")],
        output=TaskOutput(column=CLASS, kind="classification"), input_fields=["text"],
    ),
    _task(
        "tabular_classification", "Tabular Classification", "tabular", "stable", _RECORD, _CLASSIFY,
        [_col(CLASS, "label"), _col(SPLIT, "split")],
        output=TaskOutput(column=CLASS, kind="classification"),
    ),
    _task(
        "tabular_regression", "Tabular Regression", "tabular", "stable", _RECORD,
        AnnotationSpec(type="classification", requires_label_classes=False),
        [_col("target", "label"), _col(SPLIT, "split")],
        output=TaskOutput(column="target", kind="regression"),
    ),
    _task(
        "audio_classification", "Audio Classification", "audio", "stable",
        ItemSpec(payload="file", accept=_AUDIO_MIME), _CLASSIFY,
        [_col("audio_path", "storage_uri"), _col(CLASS, "label"), _col(SPLIT, "split")],
        output=TaskOutput(column=CLASS, kind="classification"),
    ),
    # -- Tier 2: experimental (sequence output) --
    _task(
        "token_classification", "Token Classification", "text", "experimental", _INLINE,
        AnnotationSpec(type="token_tags", requires_label_classes=True),
        [_col("text", "inline_text"), _col("tags", "label"), _col(SPLIT, "split")],
        output=TaskOutput(column="tags", kind="tokens"), input_fields=["text"],
    ),
    _llm_text_task("text_generation", "Text Generation", ["prompt"], "completion"),
    _llm_text_task("summarization", "Summarization", ["document"], "summary"),
    _llm_text_task("sequence_to_sequence", "Sequence to Sequence", ["source"], "target"),
    _llm_text_task("question_answering", "Question Answering", ["context", "question"], "answer"),
    # -- Tier 3: experimental (file input + free-text output) --
    _file_caption_task("image_captioning", "Image Captioning", "vision", IMAGE_PATH, "caption", _IMAGE_MIME),
    _file_caption_task("audio_captioning", "Audio Captioning", "audio", "audio_path", "caption", _AUDIO_MIME),
    _file_caption_task(
        "automatic_speech_recognition", "Automatic Speech Recognition", "audio", "audio_path", "transcript",
        _AUDIO_MIME,
    ),
    # -- Planned: no annotation/snapshot story yet --
    _planned("object_detection", "Object Detection", "vision", "bounding_box", True),
    _planned("image_segmentation", "Image Segmentation", "vision", "segmentation_mask", True),
    _planned("audio_segmentation", "Audio Segmentation", "audio", "segmentation_mask", True),
    _planned("tabular_clustering", "Tabular Clustering", "tabular", "classification", False),
    _planned("tabular_anomaly_detection", "Tabular Anomaly Detection", "tabular", "classification", False),
    _planned("text_embedding", "Text Embedding", "text", "classification", False),
]  # fmt: skip

TASK_REGISTRY: dict[str, TaskDescriptor] = {t.id: t for t in _TASKS}


# -- Helpers (ported from tasks/index.ts) ---------------------------------------------------------


def get_task_descriptor(task: str) -> TaskDescriptor:
    return TASK_REGISTRY[task]


def task_to_modality(task: str) -> DatasetModality:
    return get_task_descriptor(task).modality


def is_classification_task(task: str | None) -> bool:
    return bool(task) and get_task_descriptor(task).annotation.requires_label_classes  # type: ignore[arg-type]


def list_selectable_tasks() -> list[TaskDescriptor]:
    """Tasks Theseus can represent at all. Whether one is trainable *right now* additionally
    depends on which backend plugins are installed and available (see backends.registry)."""
    return [t for t in TASK_REGISTRY.values() if t.status != "planned"]


def get_inference_input_spec(task: str) -> dict[str, Any]:
    """What an inference request must supply for this task.

    For inline_text tasks the field names come from `input_fields`, so a multi-input task like
    question_answering reports both `context` and `question`.
    """
    d = get_task_descriptor(task)
    if d.item_spec.payload == "file":
        return {"kind": "file", "accept": d.item_spec.accept}
    if d.item_spec.payload == "record":
        return {"kind": "record"}
    if not d.input_fields:
        raise ValueError(f"Task {task!r} has no input fields to build an inference spec from")
    return {"kind": "text", "fields": d.input_fields}


def get_inference_output_kind(task: str) -> InferenceOutputKind:
    """Shape of an inference response. Must stay in sync with `LoadedModel.to_output`."""
    d = get_task_descriptor(task)
    if d.output is None:
        raise ValueError(f"Task {task!r} has no output kind defined (status: {d.status})")
    return d.output.kind


def registry_json() -> dict[str, Any]:
    """The frontend view of the registry (written to schema/task_registry.json).

    The two inference values are pre-evaluated so the browser needs no logic of its own.
    """
    tasks: dict[str, Any] = {}
    for task_id, d in TASK_REGISTRY.items():
        entry = d.model_dump(by_alias=True, exclude_none=True)
        if d.output is not None:
            entry["inferenceInputSpec"] = get_inference_input_spec(task_id)
            entry["inferenceOutputKind"] = get_inference_output_kind(task_id)
        tasks[task_id] = entry
    return {"tasks": tasks}
