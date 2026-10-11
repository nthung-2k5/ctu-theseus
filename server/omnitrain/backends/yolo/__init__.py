"""The Ultralytics YOLO trainer backend (`id = "yolo"`).

Submodules:
  tasks.py      per-task model catalogs and defaults (no ultralytics import: pure data)
  compile.py    task + snapshot + hyperparameters -> training config dict (no ultralytics import)
  runtime.py    where Ultralytics keeps its files, and one-time setup (imports ultralytics only inside functions)
  data.py       snapshot parquet -> the local folder tree Ultralytics trains on (pandas only)
  train.py      `YOLO.train()`, the progress/abort callbacks, uploading the results
  model.py      YoloLoadedModel: predict/to_output/golden_prediction over a loaded `YOLO`
  evaluate.py   the evaluation report (confusion matrix, per-class stats, top errors)
  artifacts.py  ONNX conversion
  manifest.py   preprocessing.json for a bare model export

Only tasks.py, compile.py, runtime.py, manifest.py and artifacts.py are imported at module level here, and
none of them imports ultralytics or torch at module level, so listing this backend (`available()`, `models()`,
a config-compile preview) never needs a GPU runtime and never makes Ultralytics write its settings file. The
other submodules import ultralytics/torch (directly or through those), so they are imported lazily, inside the
classmethods that need them.

Licence: Ultralytics is AGPL-3.0 (or a commercial licence from Ultralytics).
"""

import importlib.util
from typing import Any

import pandas as pd

from omnitrain.backends.base import (
    CustomModelRef,
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from omnitrain.backends.yolo.artifacts import ARTIFACTS
from omnitrain.backends.yolo.compile import YoloHyperparameters, compile_yolo_config
from omnitrain.backends.yolo.compile import hyperparameter_specs as _hyperparameter_specs
from omnitrain.backends.yolo.manifest import METADATA_FILENAME
from omnitrain.backends.yolo.tasks import YOLO_TASKS
from omnitrain.schemas.common import ParamSpec
from omnitrain.services.task_registry import SnapshotContext, TaskDescriptor


class YoloBackend(TrainerBackend):
    id = "yolo"
    label = "Ultralytics YOLO"
    description = "Fast, compact YOLO11 models fine-tuned from pretrained weights."
    Hyperparameters = YoloHyperparameters
    artifacts = ARTIFACTS
    metadata_filename = METADATA_FILENAME

    @classmethod
    def available(cls) -> str | None:
        # find_spec rather than an import: importing ultralytics creates its settings file as a side effect.
        if importlib.util.find_spec("ultralytics") is None:
            return "ultralytics is not installed"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id in YOLO_TASKS

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        spec = YOLO_TASKS.get(task.id)
        if spec is None:
            return []
        return [ModelChoice(id=m.id, label=m.label, description=m.description, pretrained=True) for m in spec.models]

    @classmethod
    def hyperparameter_specs(cls, task: TaskDescriptor) -> list[ParamSpec]:
        return _hyperparameter_specs(task)

    @classmethod
    def compile(
        cls,
        task: TaskDescriptor,
        ctx: SnapshotContext,
        hp: YoloHyperparameters,
        custom: CustomModelRef | None = None,
    ) -> dict[str, Any]:
        return compile_yolo_config(task, ctx, hp, custom)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from omnitrain.backends.yolo.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from omnitrain.backends.yolo.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from omnitrain.backends.yolo.evaluate import evaluate as _evaluate
        from omnitrain.backends.yolo.model import YoloLoadedModel

        assert isinstance(model, YoloLoadedModel)
        return _evaluate(model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        from omnitrain.backends.yolo.artifacts import convert as _convert
        from omnitrain.backends.yolo.model import YoloLoadedModel

        assert isinstance(model, YoloLoadedModel)
        return _convert(model, artifact_id, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from omnitrain.backends.yolo.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
