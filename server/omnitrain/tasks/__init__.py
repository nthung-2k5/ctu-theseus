"""Task plugins: one class per trainable (or planned) task. See `base.Task` and `builtin/`."""

from omnitrain.tasks.base import (
    TASKS,
    AnnotationSpec,
    ColumnKind,
    ColumnSpec,
    ItemSpec,
    Task,
    TaskDescriptor,
    TaskOutput,
    TaskOutputKind,
)

__all__ = [
    "TASKS",
    "AnnotationSpec",
    "ColumnKind",
    "ColumnSpec",
    "ItemSpec",
    "Task",
    "TaskDescriptor",
    "TaskOutput",
    "TaskOutputKind",
]
