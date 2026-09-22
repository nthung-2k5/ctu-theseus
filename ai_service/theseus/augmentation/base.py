"""The augmentation plugin base class.

To add an augmentation: create a module in theseus/augmentation/ops/ with a subclass that sets
`id`, `label`, `modality` and a `Params` model, and implements `apply`. Restart the backend;
GET /api/projects/{id}/augmentations returns it and the snapshot dialog renders its parameters,
with no frontend change.

Samples are plain objects per modality, and ops never mutate their input:
    vision  -> PIL.Image.Image      text    -> str
    audio   -> AudioClip            tabular -> dict[str, Any] (the item's features_json)
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from theseus.db.enums import DatasetModality
from theseus.plugins import Registry
from theseus.services.task_registry import TaskDescriptor

_registry: Registry["Augmentation"] = Registry("augmentation", "theseus.augmentation.ops")


class ParamsModel(BaseModel):
    """Base for an op's parameters: camelCase on the wire, snake_case in Python, unknown keys rejected."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class NoParams(ParamsModel):
    """For ops with nothing to tune."""


class Augmentation(ABC):
    id: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    modality: ClassVar[DatasetModality]
    Params: ClassVar[type[ParamsModel]] = NoParams
    # Sort key inside a modality.
    order: ClassVar[int] = 100

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _registry.register(cls)

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        """Whether this op is offered for a project's task.

        Default: label-preserving classification and regression tasks of this op's modality that
        actually train. Ops that rewrite the input cannot keep spatial or sequence labels (boxes,
        masks, token tags) aligned, and generative tasks have no per-item label to copy, so those
        tasks get no augmentation unless an op opts in by overriding this.
        """
        return task.modality == cls.modality and task.status != "planned" and task.annotation.type == "classification"

    @classmethod
    def prepare(cls, samples: Sequence[Any]) -> Any:
        """Optional: compute dataset-level state (e.g. per-column statistics) once from the originals.

        The result is passed back to `apply` as `state`. Only the inline modalities (text, tabular)
        receive their originals; for file-backed ones (vision, audio) the list is empty, since
        loading every file up front would not scale.
        """
        return None

    @classmethod
    @abstractmethod
    def apply(cls, sample: Any, params: Any, rng: random.Random, state: Any = None) -> Any:
        """Return a new augmented sample. `params` is a validated instance of `cls.Params`."""
