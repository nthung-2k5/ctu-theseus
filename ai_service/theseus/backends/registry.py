"""Lookup over the trainer backend plugins in theseus/backends/."""

from dataclasses import dataclass, field

from theseus.backends.base import ModelChoice, TrainerBackend, _registry
from theseus.schemas.common import ParamSpec
from theseus.services.task_registry import TASK_REGISTRY, TaskDescriptor


def get_backend(backend_id: str) -> type[TrainerBackend]:
    """The backend class for an id. Raises KeyError('Unknown trainer backend ...') if not installed."""
    return _registry.get(backend_id)


def find_backend(backend_id: str) -> type[TrainerBackend] | None:
    return _registry.find(backend_id)


def list_backends() -> list[type[TrainerBackend]]:
    """Every installed backend, sorted by id, whether or not its optional dependencies are present."""
    return sorted(_registry.all(), key=lambda b: b.id)


def trainable_backends(task: TaskDescriptor) -> list[type[TrainerBackend]]:
    """Installed, available backends that can train this task. What the create-run UI offers."""
    return [b for b in list_backends() if b.available() is None and b.supports(task)]


@dataclass
class BackendInfo:
    """Everything the API needs to describe an installed backend; a plain container so this module
    stays free of the FastAPI/pydantic response models that live in schemas/training.py."""

    id: str
    label: str
    description: str
    available: bool
    unavailable_reason: str | None
    # Every task this backend supports, whether or not it's currently available — lets a task
    # picker with no project/task in scope yet (project creation) show a task as selectable iff
    # some installed backend could eventually train it.
    supported_tasks: list[str] = field(default_factory=list)
    models: list[ModelChoice] = field(default_factory=list)
    # The hyperparameters key `models` selects (camelCase on the wire): HyperparamsBase.model_id's
    # own alias, which a backend may override (Ludwig keeps the pre-existing "encoderId" rather
    # than the generated "modelId" — see LudwigHyperparameters). A generic create-run/create-sweep
    # form must read this rather than assume "modelId", since it varies per backend.
    model_param_name: str = "modelId"
    params: list[ParamSpec] = field(default_factory=list)


def _model_param_name(backend: type[TrainerBackend]) -> str:
    field_info = backend.Hyperparameters.model_fields["model_id"]
    return field_info.alias or "modelId"


def describe(backend: type[TrainerBackend], task: TaskDescriptor | None = None) -> BackendInfo:
    reason = backend.available()
    return BackendInfo(
        id=backend.id,
        label=backend.label,
        description=backend.description,
        available=reason is None,
        unavailable_reason=reason,
        supported_tasks=[t.id for t in TASK_REGISTRY.values() if backend.supports(t)],
        models=backend.models(task) if task is not None else [],
        model_param_name=_model_param_name(backend),
        params=backend.hyperparameter_specs(task) if task is not None else [],
    )
