"""The preprocessing plugin base class.

To add a preprocessing op: create a module in theseus/preprocessing/ops/ with a subclass that sets
`id`, `label`, `modality` and a `Params` model, and implements `apply`. Restart the backend;
GET /api/projects/{id}/preprocessing returns it and the snapshot dialog renders its parameters,
with no frontend change.

Unlike augmentation, preprocessing ops are deterministic (no rng) and each is scoped to whichever
splits (train / validation / test) the snapshot config selects for it. A preprocessed item
REPLACES its original in the snapshot's membership for exactly those splits: nothing is added,
unlike augmentation's extra train-split copies (see services/preprocessing.py).

Samples are plain objects per modality, and ops never mutate their input:
    vision  -> PIL.Image.Image      text    -> str
    audio   -> AudioClip            tabular -> dict[str, Any] (the item's features_json)
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

# Reused as-is: a plugin's tunable-parameters wire model has nothing augmentation-specific about it.
from theseus.augmentation.base import NoParams, ParamsModel
from theseus.db.enums import DatasetModality
from theseus.plugins import Registry
from theseus.services.task_registry import TaskDescriptor

_registry: Registry["Preprocessing"] = Registry("preprocessing", "theseus.preprocessing.ops")


class Preprocessing(ABC):
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

        Same rule as augmentation ops (see `Augmentation.supports`): a transform that rewrites the
        input cannot keep spatial or sequence labels (boxes, masks, token tags) aligned, and
        generative tasks have no per-item label to copy, so only label-preserving classification
        and regression tasks of this op's modality get preprocessing, unless an op opts in by
        overriding this.
        """
        return task.modality == cls.modality and task.status != "planned" and task.annotation.type == "classification"

    @classmethod
    def fit(cls, samples: Sequence[Any]) -> Any:
        """Optional: compute dataset-level state (e.g. per-column mean/std) once from the TRAIN split.

        The result is passed back to `apply` as `state`. Only the inline modalities (text, tabular)
        receive samples; for file-backed ones (vision, audio) the list is empty, since loading every
        file up front would not scale (mirrors `Augmentation.prepare`).
        """
        return None

    @classmethod
    @abstractmethod
    def apply(cls, sample: Any, params: Any, state: Any = None) -> Any:
        """Return a new, preprocessed sample. `params` is a validated instance of `cls.Params`.

        Deterministic: no rng, so the same input always produces the same output.
        """
