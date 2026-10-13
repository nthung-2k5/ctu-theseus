"""The task plugin base: what a task is, independent of any training framework.

A task is one subclass of `Task` in `omnitrain/tasks/builtin/` (or any other module dropped in that
package): set `id`, `label`, `modality`, `item_spec`, `annotation`, and, unless it is `planned`, the snapshot
`columns` and the `output` a model for it predicts. It registers itself at class-creation time and shows up
after a backend restart; `projects.task` is free text, so there is no enum or migration to touch.

This is the data side only: what a dataset item looks like, its ground truth, the snapshot columns the
builder emits and the inference contract. How a task is *trained* is a trainer backend's business
(`omnitrain/backends/`), and whether it is trainable right now depends on which backends are installed
(`backends.registry.trainable_backends`). `status` only separates tasks OmniTrain can represent at all
("planned" ones have no annotation/columns story yet) from ones that are just waiting on a backend.

The closed vocabularies a task is described with (`modality`, `annotation.type`, column kinds, output kinds)
stay closed on purpose: preprocessing/augmentation ops, annotation editors and the snapshot builder are keyed
on them, so a new value is a code change in those places, not something a task plugin can introduce alone.
"""

from typing import ClassVar, Literal

from omnitrain.db.enums import AnnotationType, DatasetModality
from omnitrain.plugins import Registry
from omnitrain.schemas.common import ApiModel

ColumnKind = Literal[
    "storage_uri", "inline_text", "label", "text_sequence_label", "boxes", "split", "split_index", "scalar", "item_id"
]
TaskOutputKind = Literal["classification", "regression", "text", "tokens", "detections"]
TaskStatus = Literal["stable", "experimental", "planned"]


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
    """The immutable, serialisable view of a `Task`: what every consumer (and the web) reads."""

    id: str
    label: str
    modality: DatasetModality
    status: TaskStatus
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


TASKS: Registry["Task"] = Registry("task", "omnitrain.tasks.builtin")

_REQUIRED = ("label", "modality", "item_spec", "annotation")


class Task:
    id: ClassVar[str]
    label: ClassVar[str]
    modality: ClassVar[DatasetModality]
    status: ClassVar[TaskStatus] = "experimental"
    item_spec: ClassVar[ItemSpec]
    annotation: ClassVar[AnnotationSpec]
    columns: ClassVar[list[ColumnSpec]] = []
    input_fields: ClassVar[list[str]] = []
    output: ClassVar[TaskOutput | None] = None
    # Display order in the task picker (ascending); ties fall back to registration order. The built-ins
    # use 10, 20, ...; a plugin that does not care sorts after all of them.
    order: ClassVar[int] = 1000

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass that does not declare its own `id` is a helper base for other tasks, not a task.
        if "id" not in cls.__dict__:
            return
        cls._validate()
        TASKS.register(cls)

    @classmethod
    def _validate(cls) -> None:
        missing = [name for name in _REQUIRED if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"Task {cls.id!r} ({cls.__module__}) is missing: {', '.join(missing)}")
        if cls.status == "planned":
            if cls.output is not None or cls.columns:
                raise TypeError(f"Planned task {cls.id!r} must not define columns or an output yet")
        elif cls.output is None or not cls.columns:
            raise TypeError(f"Task {cls.id!r} is {cls.status} so it needs snapshot columns and an output")

    @classmethod
    def descriptor(cls) -> TaskDescriptor:
        return TaskDescriptor(
            id=cls.id,
            label=cls.label,
            modality=cls.modality,
            status=cls.status,
            item_spec=cls.item_spec,
            annotation=cls.annotation,
            columns=list(cls.columns),
            input_fields=list(cls.input_fields),
            output=cls.output,
        )
