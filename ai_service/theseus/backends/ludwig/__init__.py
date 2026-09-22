"""The Ludwig trainer backend: the framework every task in this repo has trained on so far, now
expressed as a `TrainerBackend` plugin instead of being wired directly into jobs/services.

Submodules:
  tasks.py      per-task Ludwig feature/encoder/knob tables (no ludwig import: pure data)
  compile.py    task + snapshot + hyperparameters -> Ludwig config dict (no ludwig import)
  train.py      LudwigModel construction, the progress/abort callback, model.train()
  model.py      LudwigLoadedModel: predict/to_output/golden_prediction over a loaded LudwigModel
  evaluate.py   the evaluation report (confusion matrix, per-class stats, top errors)
  artifacts.py  ONNX / TorchScript-export conversion
  manifest.py   preprocessing.json decompilation for a bare model export

Only tasks.py and compile.py are imported at module level here: they have no ludwig/torch import
of their own, so listing this backend (`available()`, `models()`, `compile()` for a config-compile
preview) never requires a GPU runtime. Every other submodule imports `ludwig`/`torch` at ITS module
level, so they're imported lazily, inside the classmethods that need them.
"""

from typing import Any

import pandas as pd

from theseus.backends.base import (
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from theseus.backends.ludwig.artifacts import ARTIFACTS
from theseus.backends.ludwig.compile import LudwigHyperparameters, compile_ludwig_config
from theseus.backends.ludwig.compile import hyperparameter_specs as _hyperparameter_specs
from theseus.backends.ludwig.manifest import METADATA_FILENAME
from theseus.backends.ludwig.tasks import LUDWIG_TASKS
from theseus.schemas.common import ParamSpec
from theseus.services.task_registry import SnapshotContext, TaskDescriptor


class LudwigBackend(TrainerBackend):
    id = "ludwig"
    label = "Ludwig"
    description = "Declarative deep-learning training (torch-backed): the default for stable and experimental tasks."
    Hyperparameters = LudwigHyperparameters
    artifacts = ARTIFACTS
    log_namespaces = ("ludwig",)
    metadata_filename = METADATA_FILENAME

    @classmethod
    def available(cls) -> str | None:
        try:
            import ludwig  # noqa: F401
        except ImportError as e:
            return f"ludwig is not installed ({e})"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id in LUDWIG_TASKS

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        spec = LUDWIG_TASKS.get(task.id)
        if spec is None:
            return []
        return [ModelChoice(id=e.id, label=e.label, pretrained=e.pretrained) for e in spec.encoders]

    @classmethod
    def hyperparameter_specs(cls, task: TaskDescriptor) -> list[ParamSpec]:
        return _hyperparameter_specs(task)

    @classmethod
    def compile(cls, task: TaskDescriptor, ctx: SnapshotContext, hp: LudwigHyperparameters) -> dict[str, Any]:
        return compile_ludwig_config(task, ctx, hp)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from theseus.backends.ludwig.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from theseus.backends.ludwig.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from theseus.backends.ludwig.evaluate import evaluate as _evaluate
        from theseus.backends.ludwig.model import LudwigLoadedModel

        assert isinstance(model, LudwigLoadedModel)
        return _evaluate(model._model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        from theseus.backends.ludwig.artifacts import convert as _convert
        from theseus.backends.ludwig.model import LudwigLoadedModel

        assert isinstance(model, LudwigLoadedModel)
        return _convert(model._model, artifact_id, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from theseus.backends.ludwig.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
