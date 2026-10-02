"""Lookup, parameter introspection and config validation over the preprocessing plugins."""

from pydantic import ValidationError

from theseus.params import param_specs as _param_specs
from theseus.preprocessing.base import Preprocessing, _registry
from theseus.preprocessing.config import ParamSpec, PreprocessingConfig, PreprocessingInfo
from theseus.services import plugin_settings
from theseus.services.task_registry import TaskDescriptor

_MODALITY_ORDER = ("vision", "text", "audio", "tabular")


class PreprocessingConfigError(ValueError):
    """A snapshot preprocessing request that cannot be honoured; the message is safe to show the user."""


def get_preprocessing(op_id: str) -> type[Preprocessing]:
    return _registry.get(op_id)


def list_preprocessing(
    task: TaskDescriptor | None = None, *, include_disabled: bool = False
) -> list[type[Preprocessing]]:
    """Installed ops, ordered for display; only those supporting `task` when given, and not switched off by an admin."""
    ops = [
        op
        for op in _registry.all()
        if (task is None or op.supports(task))
        and (include_disabled or plugin_settings.is_enabled("preprocessing", op.id, task.id if task else None))
    ]
    return sorted(ops, key=lambda op: (_MODALITY_ORDER.index(op.modality), op.order, op.id))


def param_specs(op: type[Preprocessing]) -> list[ParamSpec]:
    return _param_specs(op.Params)


def describe(op: type[Preprocessing]) -> PreprocessingInfo:
    return PreprocessingInfo(
        id=op.id, label=op.label, description=op.description, modality=op.modality, params=param_specs(op)
    )


# -- Validation ------------------------------------------------------------------------------


def validate_config(task: TaskDescriptor, config: PreprocessingConfig) -> PreprocessingConfig:
    """Check every op exists, supports the task and has valid params; return the config with params normalised.

    Normalising fills defaults, so the config stored on the snapshot records exactly what was applied.
    """
    seen: set[str] = set()
    ops = []
    for op_config in config.ops:
        op = _registry.find(op_config.id)
        if op is None:
            raise PreprocessingConfigError(f"Unknown preprocessing op '{op_config.id}'")
        if not op.supports(task):
            raise PreprocessingConfigError(f"Preprocessing op '{op.id}' is not available for {task.label}")
        if not plugin_settings.is_enabled("preprocessing", op.id, task.id):
            raise PreprocessingConfigError(f"Preprocessing op '{op.id}' has been disabled by an administrator")
        if op.id in seen:
            raise PreprocessingConfigError(f"Preprocessing op '{op.id}' is listed more than once")
        seen.add(op.id)
        try:
            params = op.Params.model_validate(op_config.params)
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise PreprocessingConfigError(f"Invalid parameter '{where}' for '{op.id}': {first['msg']}") from None
        ops.append(op_config.model_copy(update={"params": params.model_dump(by_alias=True)}))
    return config.model_copy(update={"ops": ops, "skipped_items": None})
