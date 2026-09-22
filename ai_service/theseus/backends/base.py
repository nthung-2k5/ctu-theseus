"""The trainer backend plugin base class.

To add a backend: create a package in theseus/backends/ with a subclass that sets `id`/`label`, a
`Hyperparameters` model, and implements `supports`/`models`/`compile`/`train`/`load`/`convert`.
Restart the backend; `GET /api/training-backends` returns it and the web renders its models and
hyperparameters, with no enum, migration or frontend change (beyond installing the plugin's own
optional dependency).

Heavy ML dependencies (torch, ludwig, sklearn, ...) MUST NOT be imported at module level here or
in a subclass's own top-level module: listing installed backends (`backends.registry.list_backends()`)
imports every plugin package just to read `id`/`label`/`available()`, and the API process must be
importable without a GPU runtime present. Import heavy dependencies inside the classmethod bodies
that need them (see `theseus/backends/ludwig/__init__.py` for the pattern).

Two invariants every backend must honor, carried over from the Ludwig-only code this replaces:

  * `OutputSpec.labels` (classification class names, in the model's OWN index order) must come
    from the trained model artifact, never from the `label_classes` Postgres table: Postgres has
    no idea what index order a training run assigned to each class, and guessing wrong silently
    mislabels every prediction.
  * A backend's `train()` reports a `validation`-split `loss` metric (lower is better) wherever it
    can: `events/writer.py` uses the lowest validation loss so far as a task-agnostic definition of
    a run's best epoch. A backend with nothing loss-like to report simply omits it; best_epoch then
    stays null.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from theseus.plugins import Registry
from theseus.services.predict import InferenceOutput
from theseus.services.task_registry import SnapshotContext, TaskDescriptor

_registry: Registry["TrainerBackend"] = Registry("trainer backend", "theseus.backends")


class ConfigError(ValueError):
    """A user-correctable problem with the requested training configuration."""


class HyperparamsBase(BaseModel):
    """camelCase on the wire, unknown keys rejected. Every backend's `Hyperparameters` subclasses
    this, adding whatever extra knobs its own models support."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    # Model choice, from this backend's `models(task)` list. Defaults to the first when unset.
    model_id: str | None = None
    epochs: int | None = Field(default=None, gt=0, le=500)
    batch_size: int | Literal["auto"] | None = None
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    early_stop_patience: int | None = Field(default=None, ge=-1)


class ModelChoice(BaseModel):
    """One selectable model/architecture within a backend, for a given task."""

    id: str
    label: str
    description: str = ""
    pretrained: bool = False


@dataclass(frozen=True)
class Artifact:
    """One exportable, converted-model file a backend can produce from a trained run."""

    id: str
    # File name written inside the export directory, and the S3 object name under the run prefix.
    filename: str


# Shared filename convention for the artifact ids every backend is expected to know how to name
# consistently (a backend need not produce all, or only, these — see its own `artifacts` dict).
# `ExportFormat.assemble` (export/bundle.py) lays out a bundle by artifact filename alone and never
# needs to know which backend produced the model inside it, so the convention lives here rather
# than duplicated per backend.
ARTIFACT_FILENAMES: dict[str, str] = {"onnx": "model.onnx", "torch_export": "model.pt2"}


OutputKind = Literal["classification", "regression", "text", "tokens"]


@dataclass
class OutputSpec:
    name: str
    kind: OutputKind
    # Class names in the model's own index order. None for non-classification outputs.
    labels: list[str] | None = None


class LoadedModel(ABC):
    """A trained model resident in memory: the seam `services/model_cache.py`, `jobs/inference.py`,
    `jobs/export.py` and `services/evaluate.py` use instead of talking to a specific framework."""

    input_columns: list[str]
    output: OutputSpec

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Raw framework prediction columns (`{feature}_predictions`, `{feature}_probabilities`, ...),
        one row per input row. Consumed directly for batch CSV results, and via `to_output`/
        `golden_prediction` for the single-request/export-verification shapes."""

    @abstractmethod
    def to_output(
        self, predictions: pd.DataFrame, *, top_k: int = 100, input_tokens: list[str] | None = None
    ) -> InferenceOutput:
        """Row 0 of `predictions` as the tagged-union inference response shape
        (`InferenceResponseSchema.output`: classification/regression/text/tokens)."""

    @abstractmethod
    def golden_prediction(self, predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, float]:
        """Row 0 of `predictions` as a flat `{class_or_feature: value}` dict, `>= threshold`, sorted
        descending. This is the shape `expected.json` embeds in every export bundle; every generated
        devkit/app client parses it, so its shape is part of the export contract and must not change."""

    def close(self) -> None:  # noqa: B027
        """Release native/GPU resources held by this model. Default: nothing to release."""


@dataclass
class EvalResult:
    report: dict[str, Any]
    # Full per-row predictions, for the accompanying predictions.parquet download.
    predictions: pd.DataFrame


@dataclass
class TrainContext:
    """What `jobs/train.py` hands a backend's `train()`.

    Wraps the event-writer and abort machinery so a backend module never has to import
    `theseus.events` or `theseus.jobs.abort` itself. `heartbeat`/`check_abort` are safe to call
    from the training thread (as a Ludwig `Callback` does); `report` is too.
    """

    run_id: str
    config: dict[str, Any]
    # s3fs paths (see services/storage.s3fs_path), readable/writable by pandas/the framework directly.
    dataset_uri: str
    output_uri: str
    workdir: str
    _heartbeat: Callable[[], None]
    _check_abort: Callable[[], None]
    _report: Callable[[int, str, dict[str, float]], None]

    def heartbeat(self) -> None:
        self._heartbeat()

    def check_abort(self) -> None:
        """Raises `theseus.jobs.abort.TrainingAborted` if the run has been asked to cancel."""
        self._check_abort()

    def report(self, epoch: int, split: str, metrics: dict[str, float]) -> None:
        self._report(epoch, split, metrics)


class TrainerBackend(ABC):
    id: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str] = ""
    Hyperparameters: ClassVar[type[HyperparamsBase]] = HyperparamsBase
    # Exportable artifacts this backend can convert a trained model into, keyed by Artifact.id.
    artifacts: ClassVar[dict[str, Artifact]] = {}
    # Logger namespaces streamed into the run's live log / log file (see events/log_handler.py).
    log_namespaces: ClassVar[tuple[str, ...]] = ()
    # Suffix of an S3 object under the run's results prefix that `preprocessing_manifest` needs
    # beyond the compiled config, if any (e.g. Ludwig's "training_set_metadata.json").
    metadata_filename: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _registry.register(cls)

    @classmethod
    def available(cls) -> str | None:
        """None if this backend is ready to use; otherwise a user-facing reason it is not (e.g. a
        missing optional dependency). The default assumes any import succeeded, i.e. always available."""
        return None

    @classmethod
    @abstractmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        """Whether this backend can train the given task at all."""

    @classmethod
    @abstractmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        """Selectable models/architectures for this task. Empty is valid (a backend with one
        fixed model per task, e.g. a classical ML backend with no pretrained-weights choice)."""

    @classmethod
    @abstractmethod
    def compile(cls, task: TaskDescriptor, ctx: SnapshotContext, hp: HyperparamsBase) -> dict[str, Any]:
        """Turn a task descriptor, snapshot context and validated hyperparameters into this
        backend's own training config (opaque outside the backend; stored verbatim as
        `training_runs.config`). Raises `ConfigError` for a user-correctable problem."""

    @classmethod
    @abstractmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        """Train synchronously (runs in a worker thread/executor). Must call `run.check_abort()` at
        any point it is safe to stop, and should call `run.report(...)` with progress. Writes the
        trained model under `run.output_uri` and returns it already loaded (so the caller can
        evaluate it immediately, without a reload from storage). Raise on failure; raise
        `abort.TrainingAborted` (via `run.check_abort()`) to stop early."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str) -> LoadedModel:
        """Load a trained model from a local directory (already downloaded from `run.output_uri`)."""

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        """Build the evaluation report + predictions download for a trained model against `df`
        (typically the test split, see `services/evaluate.pick_eval_split`). Returns None if there
        is nothing to evaluate. Default: no evaluation report support."""
        return None

    @classmethod
    @abstractmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        """Write `cls.artifacts[artifact_id]` to a file under `workdir` and return its path."""

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        """The `preprocessing.json` a bare model export needs to be usable: exact preprocessing
        parameters (image resize/normalize, tokenizer params, ...) and, for classification outputs,
        the class list in the model's own index order. `meta` is the parsed contents of the S3
        object named by `metadata_filename`, or None if `metadata_filename` is unset or the object
        was not found. The default (no metadata file, nothing to add beyond the compiled config)
        raises NotImplementedError; the export README then tells the user so."""
        raise NotImplementedError


__all__ = [
    "ARTIFACT_FILENAMES",
    "Artifact",
    "ConfigError",
    "EvalResult",
    "HyperparamsBase",
    "LoadedModel",
    "ModelChoice",
    "OutputKind",
    "OutputSpec",
    "TrainContext",
    "TrainerBackend",
]
