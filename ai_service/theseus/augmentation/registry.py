"""Lookup, parameter introspection and config validation over the augmentation plugins."""

import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import ValidationError
from pydantic.alias_generators import to_camel
from pydantic.fields import FieldInfo

from theseus.augmentation.base import Augmentation, _registry
from theseus.augmentation.config import AugmentationConfig, AugmentationInfo, ParamSpec
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


# -- Parameter introspection -----------------------------------------------------------------


def _bound(field: FieldInfo, *names: str) -> float | None:
    for meta in field.metadata:
        for name in names:
            value = getattr(meta, name, None)
            if value is not None:
                return float(value)
    return None


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _default_step(lo: float | None, hi: float | None) -> float:
    span = (hi - lo) if lo is not None and hi is not None else 1.0
    return 0.01 if span <= 1 else 0.1 if span <= 20 else 1.0


def _humanize(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _param_spec(name: str, field: FieldInfo) -> ParamSpec:
    annotation = _unwrap_optional(field.annotation)
    label = field.title or _humanize(name)
    common = {"name": to_camel(name), "label": label, "description": field.description, "default": field.default}
    if annotation is bool:
        return ParamSpec(type="bool", **common)
    if get_origin(annotation) is Literal:
        return ParamSpec(type="choice", choices=[str(c) for c in get_args(annotation)], **common)
    if annotation in (int, float):
        lo, hi = _bound(field, "ge", "gt"), _bound(field, "le", "lt")
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        step = extra.get("step") or (1.0 if annotation is int else _default_step(lo, hi))
        return ParamSpec(type="int" if annotation is int else "float", min=lo, max=hi, step=step, **common)
    raise TypeError(f"Augmentation parameter {name!r} has unsupported type {annotation!r}")


def param_specs(op: type[Augmentation]) -> list[ParamSpec]:
    return [_param_spec(name, field) for name, field in op.Params.model_fields.items()]


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
