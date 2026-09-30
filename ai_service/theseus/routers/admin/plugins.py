"""/api/admin/plugins: list every plugin, built-in model and task with its switch, and flip them.

Nothing here changes what is installed (the registries are immutable until restart); it only writes
the overrides that services/plugin_settings.py applies when listing and validating.
"""

from collections.abc import Callable, Iterable

from fastapi import APIRouter, HTTPException

from theseus.augmentation.registry import list_augmentations
from theseus.backends.registry import list_backends
from theseus.deps import AdminId, SessionDep
from theseus.export.registry import list_export_formats
from theseus.preprocessing.registry import list_preprocessing
from theseus.schemas.admin import PluginEntry, PluginListResponse, PluginResponse, PluginTaskState, SetPluginBody
from theseus.services import plugin_settings as ps
from theseus.services.task_registry import TASK_REGISTRY, TaskDescriptor

router = APIRouter(prefix="/plugins")


def _entry(
    kind: str,
    plugin_id: str,
    *,
    label: str,
    tasks: Iterable[TaskDescriptor],
    description: str = "",
    group: str | None = None,
    available: bool = True,
    unavailable_reason: str | None = None,
) -> PluginEntry:
    overrides = ps.snapshot()
    return PluginEntry(
        kind=kind,
        id=plugin_id,
        label=label,
        description=description,
        group=group,
        available=available,
        unavailable_reason=unavailable_reason,
        enabled=ps.is_enabled(kind, plugin_id),
        tasks=[
            PluginTaskState(
                task=t.id,
                task_label=t.label,
                enabled=ps.is_enabled(kind, plugin_id, t.id),
                overridden=(kind, plugin_id, t.id) in overrides,
            )
            for t in tasks
        ],
    )


def _supporting(supports: Callable[[TaskDescriptor], bool]) -> list[TaskDescriptor]:
    return [t for t in TASK_REGISTRY.values() if supports(t)]


def _all_entries() -> list[PluginEntry]:
    entries: list[PluginEntry] = []

    for backend in list_backends():
        reason = backend.available()
        entries.append(
            _entry(
                "backend",
                backend.id,
                label=backend.label,
                description=backend.description,
                tasks=_supporting(backend.supports),
                available=reason is None,
                unavailable_reason=reason,
            )
        )

    # A built-in model is offered per task, so gather the tasks each one appears for.
    for backend in list_backends():
        models: dict[str, tuple[str, list[TaskDescriptor]]] = {}
        for task in _supporting(backend.supports):
            for m in backend.models(task):
                models.setdefault(m.id, (m.label, []))[1].append(task)
        for model_id, (label, tasks) in models.items():
            entries.append(
                _entry(
                    "builtin_model",
                    ps.builtin_model_key(backend.id, model_id),
                    label=label,
                    tasks=tasks,
                    group=backend.id,
                )
            )

    for fmt in list_export_formats(include_disabled=True):
        entries.append(
            _entry(
                "export_format",
                fmt.id,
                label=fmt.label,
                description=fmt.description,
                group=fmt.group,
                tasks=_supporting(fmt.supports),
            )
        )
    for op in list_preprocessing(include_disabled=True):
        entries.append(
            _entry(
                "preprocessing",
                op.id,
                label=op.label,
                description=op.description,
                group=op.modality,
                tasks=_supporting(op.supports),
            )
        )
    for aug in list_augmentations(include_disabled=True):
        entries.append(
            _entry(
                "augmentation",
                aug.id,
                label=aug.label,
                description=aug.description,
                group=aug.modality,
                tasks=_supporting(aug.supports),
            )
        )

    # A task has only the every-task switch; there is no "task within a task".
    for task in TASK_REGISTRY.values():
        entries.append(_entry("task", task.id, label=task.label, group=task.status, tasks=[]))
    return entries


@router.get("", response_model=PluginListResponse)
async def admin_list_plugins() -> PluginListResponse:
    return PluginListResponse(plugins=_all_entries())


@router.put("/{kind}/{plugin_id}", response_model=PluginResponse)
async def admin_set_plugin(
    kind: str, plugin_id: str, body: SetPluginBody, admin_id: AdminId, session: SessionDep
) -> PluginResponse:
    if kind not in ps.PLUGIN_KINDS:
        raise HTTPException(404, f"Unknown plugin kind '{kind}'")
    target = next((e for e in _all_entries() if e.kind == kind and e.id == plugin_id), None)
    if target is None:
        raise HTTPException(404, f"Unknown {kind} '{plugin_id}'")
    if body.task is not None and body.task not in {t.task for t in target.tasks}:
        raise HTTPException(400, f"'{plugin_id}' is not offered for task '{body.task}'")

    await ps.set_enabled(
        session, kind=kind, plugin_id=plugin_id, task=body.task, enabled=body.enabled, user_id=admin_id
    )
    updated = next(e for e in _all_entries() if e.kind == kind and e.id == plugin_id)
    return PluginResponse(plugin=updated)
