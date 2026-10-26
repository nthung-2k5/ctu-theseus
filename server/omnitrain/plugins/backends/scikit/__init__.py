"""The scikit-learn trainer backend (`id = "sklearn"`): classical models fitted in one go on a CPU.

Tabular classification and regression (gradient-boosted trees, random forest, a linear model) and text classification
(TF-IDF features with logistic regression or naive Bayes). Quick, needs no GPU, and the usual baseline a neural model
has to beat.

Submodules:
  tasks.py       per-task model catalogs (no scikit-learn import: pure data)
  compile.py     task + snapshot + hyperparameters -> training config dict (no scikit-learn import)
  estimators.py  the config -> an unfitted pipeline, and saving/loading a fitted one (skops, not pickle)
  data.py        snapshot parquet -> features and labels (pandas only)
  train.py       one `fit()`, the train/validation scores, uploading the model
  model.py       SklearnLoadedModel: predict/to_output/golden_prediction over a loaded pipeline
  evaluate.py    the evaluation report
  artifacts.py   ONNX conversion (skl2onnx), shaped to the generated clients' contract
  manifest.py    preprocessing.json for a bare model export

Only tasks.py, compile.py, artifacts.py and manifest.py are imported at module level here, and none of them imports
scikit-learn, so listing the backend never loads it.

Not offered yet: bring-your-own models, clustering, anomaly detection, cross-validation and calibration.

Licence: scikit-learn is BSD-3-Clause.
"""

import importlib.util
from typing import Any

import pandas as pd

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import (
    CustomModelRef,
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from omnitrain.plugins.backends.scikit.artifacts import ARTIFACTS
from omnitrain.plugins.backends.scikit.compile import SklearnHyperparameters, compile_sklearn_config
from omnitrain.plugins.backends.scikit.compile import hyperparameter_specs as _hyperparameter_specs
from omnitrain.plugins.backends.scikit.estimators import METADATA_FILENAME
from omnitrain.plugins.backends.scikit.tasks import SKLEARN_TASKS
from omnitrain.schemas.common import ParamSpec


class SklearnBackend(TrainerBackend):
    id = "sklearn"
    label = "scikit-learn"
    description = "Fast classical models (gradient-boosted trees, random forests, linear models) that train on a CPU."
    Hyperparameters = SklearnHyperparameters
    artifacts = ARTIFACTS
    metadata_filename = METADATA_FILENAME
    # A tree ensemble on a CPU beats a small network on most tables, trains in seconds, and fits its imputer and scaler
    # on the train split only (Ludwig's built-in normalisation learns from every row).
    default_for = frozenset({"tabular_classification", "tabular_regression"})

    @classmethod
    def available(cls) -> str | None:
        for module in ("sklearn", "skops"):
            if importlib.util.find_spec(module) is None:
                return f"{module} is not installed"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id in SKLEARN_TASKS

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        spec = SKLEARN_TASKS.get(task.id)
        return list(spec.models) if spec is not None else []

    @classmethod
    def hyperparameter_specs(cls, task: TaskDescriptor) -> list[ParamSpec]:
        return _hyperparameter_specs(task)

    @classmethod
    def compile(
        cls,
        task: TaskDescriptor,
        ctx: SnapshotContext,
        hp: SklearnHyperparameters,
        custom: CustomModelRef | None = None,
    ) -> dict[str, Any]:
        return compile_sklearn_config(task, ctx, hp, custom)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from omnitrain.plugins.backends.scikit.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from omnitrain.plugins.backends.scikit.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from omnitrain.plugins.backends.scikit.evaluate import evaluate as _evaluate
        from omnitrain.plugins.backends.scikit.model import SklearnLoadedModel

        assert isinstance(model, SklearnLoadedModel)
        return _evaluate(model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        from omnitrain.plugins.backends.scikit.artifacts import convert as _convert
        from omnitrain.plugins.backends.scikit.model import SklearnLoadedModel

        assert isinstance(model, SklearnLoadedModel)
        return _convert(model, artifact_id, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from omnitrain.plugins.backends.scikit.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
