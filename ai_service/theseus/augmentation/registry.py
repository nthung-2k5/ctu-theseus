"""Lookup, parameter introspection and config validation over the augmentation plugins."""

from pydantic import ValidationError

from theseus.augmentation.base import Augmentation, _registry
from theseus.augmentation.config import AugmentationConfig, AugmentationInfo, ParamSpec
from theseus.params import param_specs as _param_specs
from theseus.services.task_registry import TaskDescriptor

_MODALITY_ORDER = ("vision", "text", "audio", "tabular")


class AugmentationConfigError(ValueError):
    """A snapshot augmentation request that cannot be honoured; the message is safe to show the user."""


def get_augmentation(op_id: str) -> type[Augmentation]:
    return _registry.get(op_id)


def list_augmentations(task: TaskDescriptor | None = None) -> list[type[Augmentation]]:
    """Installed ops, ordered for display; only those supporting `task` when given."""
    ops = [op for op in _registry.all() if task is None or op.supports(task)]
    return sorted(ops, key=lambda op: (_MODALITY_ORDER.index(op.modality), op.order, op.id))


def param_specs(op: type[Augmentation]) -> list[ParamSpec]:
    return _param_specs(op.Params)


def describe(op: type[Augmentation]) -> AugmentationInfo:
    return AugmentationInfo(
        id=op.id, label=op.label, description=op.description, modality=op.modality, params=param_specs(op)
    )


# -- Validation ------------------------------------------------------------------------------


def validate_config(task: TaskDescriptor, config: AugmentationConfig) -> AugmentationConfig:
    """Check every op exists, supports the task and has valid params; return the config with params normalised.

    Normalising fills defaults, so the config stored on the snapshot records exactly what was applied.
    """
    seen: set[str] = set()
    ops = []
    for op_config in config.ops:
        op = _registry.find(op_config.id)
        if op is None:
            raise AugmentationConfigError(f"Unknown augmentation '{op_config.id}'")
        if not op.supports(task):
            raise AugmentationConfigError(f"Augmentation '{op.id}' is not available for {task.label}")
        if op.id in seen:
            raise AugmentationConfigError(f"Augmentation '{op.id}' is listed more than once")
        seen.add(op.id)
        try:
            params = op.Params.model_validate(op_config.params)
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise AugmentationConfigError(f"Invalid parameter '{where}' for '{op.id}': {first['msg']}") from None
        ops.append(op_config.model_copy(update={"params": params.model_dump(by_alias=True)}))
    return config.model_copy(update={"ops": ops, "skipped_items": None})
