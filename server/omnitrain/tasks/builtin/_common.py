"""Shorthands shared by the built-in task modules (underscore-prefixed, so not loaded as a task module)."""

from typing import ClassVar

from omnitrain.core import constants as C
from omnitrain.tasks.base import AnnotationSpec, ColumnKind, ColumnSpec, ItemSpec, Task, TaskOutput

IMAGE_PATH = C.IMAGE_PATH_COLUMN_NAME
CLASS = C.CLASS_COLUMN_NAME
SPLIT = C.SPLIT_COLUMN_NAME
BOXES = "boxes"
VIDEO_PATH = "video_path"
AUDIO_PATH = "audio_path"

AUDIO_MIME = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"]
IMAGE_MIME = ["image/jpeg", "image/png"]
VIDEO_MIME = ["video/mp4", "video/webm"]

CLASSIFY = AnnotationSpec(type="classification", requires_label_classes=True)
FREE_TEXT = AnnotationSpec(type="text_sequence", requires_label_classes=False)
INLINE = ItemSpec(payload="inline_text")
RECORD = ItemSpec(payload="record")


def col(name: str, kind: ColumnKind) -> ColumnSpec:
    return ColumnSpec(name=name, kind=kind)


class ClassificationTask(Task):
    """Helper base: one class per item, read from a single `source_column` (a file path or inline text)."""

    source_column: ClassVar[ColumnSpec]

    annotation = CLASSIFY

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Runs before Task.__init_subclass__ validates and registers the class.
        if "id" in cls.__dict__:
            cls.columns = [cls.source_column, col(CLASS, "label"), col(SPLIT, "split")]
            cls.output = TaskOutput(column=CLASS, kind="classification")
        super().__init_subclass__(**kwargs)


class LlmTextTask(Task):
    """Helper base: free text in (`input_fields`), free text out (`output_field`)."""

    output_field: ClassVar[str]

    modality = "text"
    item_spec = INLINE
    annotation = FREE_TEXT

    def __init_subclass__(cls, **kwargs: object) -> None:
        if "id" in cls.__dict__:
            cls.columns = [
                *(col(c, "inline_text") for c in cls.input_fields),
                col(cls.output_field, "inline_text"),
                col(SPLIT, "split"),
            ]
            cls.output = TaskOutput(column=cls.output_field, kind="text")
        super().__init_subclass__(**kwargs)


class FileCaptionTask(Task):
    """Helper base: a file in, a free-text caption or transcript out."""

    path_column: ClassVar[str]
    output_field: ClassVar[str]
    accept: ClassVar[list[str]]

    annotation = FREE_TEXT

    def __init_subclass__(cls, **kwargs: object) -> None:
        if "id" in cls.__dict__:
            cls.item_spec = ItemSpec(payload="file", accept=cls.accept)
            cls.columns = [
                col(cls.path_column, "storage_uri"),
                col(cls.output_field, "text_sequence_label"),
                col(SPLIT, "split"),
            ]
            cls.output = TaskOutput(column=cls.output_field, kind="text")
        super().__init_subclass__(**kwargs)


class PlannedTask(Task):
    """Helper base: genuinely blocked, not just unbuilt. OmniTrain has no mask annotation editor or snapshot
    columns for these yet, so no backend could support one even in principle today."""

    status = "planned"
    item_spec = ItemSpec(payload="file")
