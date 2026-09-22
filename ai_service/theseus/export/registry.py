"""Lookup over the export format plugins in theseus/export/formats/."""

from theseus.export.formats.base import ExportFormat, _registry
from theseus.services.task_registry import TaskDescriptor


def get_export_format(format_id: str) -> type[ExportFormat]:
    """The format class for an id. Raises KeyError('Unknown export format ...') if it is not installed."""
    return _registry.get(format_id)


def find_export_format(format_id: str) -> type[ExportFormat] | None:
    return _registry.find(format_id)


def list_export_formats(task: TaskDescriptor | None = None) -> list[type[ExportFormat]]:
    """Installed formats, grouped and ordered for display; only those supporting `task` when given."""
    formats = [f for f in _registry.all() if task is None or f.supports(task)]
    return sorted(formats, key=lambda f: (_group_rank(f.group), f.group, f.order, f.id))


# Known groups come first in this order; any group a plugin invents sorts alphabetically after them.
_GROUP_ORDER = ("Model", "Devkit", "App")


def _group_rank(group: str) -> int:
    return _GROUP_ORDER.index(group) if group in _GROUP_ORDER else len(_GROUP_ORDER)
