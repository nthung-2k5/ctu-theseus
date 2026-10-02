"""Flatten a pydantic model's fields into `ParamSpec`s the web can render a form from, without
knowing anything about what the model is for.

Shared by every plugin system whose UI needs "here are this thing's tunable parameters, generate
a form": augmentation ops (`theseus/augmentation/registry.py`) and trainer backend hyperparameters
(`theseus/backends/registry.py`).
"""

import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from pydantic.fields import FieldInfo

from theseus.schemas.common import ParamSpec


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
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
    common = {
        "name": to_camel(name),
        "label": label,
        "description": field.description,
        "default": field.default,
        "group": extra.get("group"),
    }
    if annotation is bool:
        return ParamSpec(type="bool", **common)
    if get_origin(annotation) is Literal:
        return ParamSpec(type="choice", choices=[str(c) for c in get_args(annotation)], **common)
    if annotation in (int, float):
        lo, hi = _bound(field, "ge", "gt"), _bound(field, "le", "lt")
        step = extra.get("step") or (1.0 if annotation is int else _default_step(lo, hi))
        return ParamSpec(type="int" if annotation is int else "float", min=lo, max=hi, step=step, **common)
    raise TypeError(f"Parameter {name!r} has unsupported type {annotation!r}")


def param_specs(model: type[BaseModel], *, strict: bool = True) -> list[ParamSpec]:
    """One `ParamSpec` per field of a pydantic model — an augmentation op's `Params`, or a trainer
    backend's `Hyperparameters`.

    `strict=True` (the default, and always used for augmentation ops) raises `TypeError` on a
    field with no sensible UI representation: an op's `Params` model is small and fully under its
    author's control, so a field that doesn't fit is a bug worth surfacing immediately. A trainer
    backend's `Hyperparameters` is different: it inherits shared fields from `HyperparamsBase`
    (e.g. `batchSize: int | Literal["auto"]`) that aren't representable as one generic `ParamSpec`,
    and `modelId` is deliberately exposed via `models(task)` instead of here. Pass `strict=False`
    (as `TrainerBackend.hyperparameter_specs`'s default implementation does) to skip those rather
    than fail; a backend whose defaults or bounds vary by task should override
    `hyperparameter_specs` instead of relying on this generic introspection at all.
    """
    specs = []
    for name, field in model.model_fields.items():
        try:
            specs.append(_param_spec(name, field))
        except TypeError:
            if strict:
                raise
    return specs
