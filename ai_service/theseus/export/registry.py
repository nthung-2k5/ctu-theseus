"""Lookup over the export format plugins in theseus/export/formats/."""

from theseus.backends.base import TrainerBackend
from theseus.export.formats.base import ExportFormat, _registry
from theseus.services.task_registry import TaskDescriptor


def get_export_format(format_id: str) -> type[ExportFormat]:
    """The format class for an id. Raises KeyError('Unknown export format ...') if it is not installed."""
    return _registry.get(format_id)


def find_export_format(format_id: str) -> type[ExportFormat] | None:
    return _registry.find(format_id)


def list_export_formats(
    task: TaskDescriptor | None = None, backend: type[TrainerBackend] | None = None
) -> list[type[ExportFormat]]:
    """Installed formats, grouped and ordered for display.

    With `task`, only formats that support it. With `backend`, additionally only formats built
    from an artifact that backend actually produces (e.g. a run trained by a backend with no
    torch_export support is never offered a torch_export-based format).
    """
    formats = [
        f
        for f in _registry.all()
        if (task is None or f.supports(task)) and (backend is None or f.artifact in backend.artifacts)
    ]
    return sorted(formats, key=lambda f: (_group_rank(f.group), f.group, f.order, f.id))


# Known groups come first in this order; any group a plugin invents sorts alphabetically after them.
_GROUP_ORDER = ("Model", "Devkit", "App")


def _group_rank(group: str) -> int:
    return _GROUP_ORDER.index(group) if group in _GROUP_ORDER else len(_GROUP_ORDER)
