"""The PyTorch trainer backend (`id = "pytorch"`): code-defined models, trained by a plain PyTorch loop.

Where Ludwig is configured (a YAML of feature types and encoders that Ludwig turns into a network), this backend is
written: the network is a torchvision model with a new head (images), a small perceptron (tables), a convolutional
network over a spectrogram (sounds) or a word-level network (texts), the loop is ours, and what a model expects as
input is one spec (`ImageSpec`, `TabularSpec`, `AudioSpec`, `TextSpec`) that training, evaluation, serving and the
export manifest all read. That is what makes a client's prediction agree with the platform's, and it removes Ludwig's
hidden preprocessing from the picture. A table's imputation and scaling are fitted on the train split only and live
inside the exported graph.

Submodules:
  tasks.py      the model catalog and defaults (pure data)
  spec.py       `ImageSpec`, `TabularSpec`, `AudioSpec` and `TextSpec`: the single definition of an input (pure data)
  compile.py    task + snapshot + hyperparameters -> training config dict (no torch import)
  runtime.py    where weights are cached, device and dataloader choice (torch imported only inside functions)
  images.py     image file -> tensor, shared by train, serve and the docs of the export clients (PIL, numpy)
  data.py       snapshot parquet -> local image files and labelled samples per split (pandas, PIL)
  audio.py      audio file -> log-mel spectrogram, the one recipe (numpy, scipy)
  text.py       text -> word ids, the one recipe: tokenizer, train-only vocabulary, padding (numpy, pandas)
  tagging.py    token classification's data: a text's whitespace words -> ids, one tag per word (numpy, pandas)
  tabular.py    snapshot parquet -> feature matrix and targets, train-only statistics (numpy, pandas)
  network.py    builds a torchvision model with a fresh head, the table perceptron, or a text model
  loop.py       heartbeats, the optimizer step (schedule, clipping, mixed precision), the objective and `fit`
  train.py      the flows (images, tables, sounds, texts), progress/abort, uploading the result
  model.py      PytorchLoadedModel (Image/Tabular/Audio/TextModel): predict/to_output/golden_prediction
  evaluate.py   the evaluation report (the shared classification and regression reports)
  artifacts.py  ONNX conversion, straight from the module
  manifest.py   preprocessing.json for a bare model export
  custom.py     bring-your-own Hugging Face models (image and text): kinds, file checks, the image size and
                normalisation, or the token limit, a model dictates (JSON only, no torch)
  hf.py         builds, freezes, saves and loads such a model as a plain `[N, 3, H, W] -> logits` or
                `[N, max_tokens] ids -> logits` module, and runs its tokenizer (torch)

Only tasks.py, spec.py, compile.py, runtime.py, manifest.py, artifacts.py and custom.py are imported at module level
here, and none of them imports torch at module level, so listing this backend never needs a GPU runtime. The others
import torch, so they are imported lazily, inside the classmethods that need them.

Licence: torchvision and its pretrained ImageNet weights are BSD-3-Clause. Not offered yet: built-in pretrained text
models (a Hugging Face encoder is brought, not bundled), a bring-your-own token classifier, bring-your-own models other than
Hugging Face image and text classifiers, categorical table columns.
"""

import importlib.util
from typing import Any

import pandas as pd

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import (
    CustomModelKind,
    CustomModelRef,
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from omnitrain.plugins.backends.pytorch import custom as _custom
from omnitrain.plugins.backends.pytorch.artifacts import ARTIFACTS
from omnitrain.plugins.backends.pytorch.compile import PytorchHyperparameters, compile_pytorch_config
from omnitrain.plugins.backends.pytorch.compile import hyperparameter_specs as _hyperparameter_specs
from omnitrain.plugins.backends.pytorch.manifest import METADATA_FILENAME
from omnitrain.plugins.backends.pytorch.tasks import (
    AUDIO_MODELS,
    AUDIO_TASKS,
    IMAGE_MODELS,
    IMAGE_TASKS,
    TABULAR_MODELS,
    TABULAR_TASKS,
    TAGGER_MODELS,
    TEXT_MODELS,
    TEXT_TASKS,
    TOKEN_TASKS,
)
from omnitrain.schemas.common import ParamSpec


class PytorchBackend(TrainerBackend):
    id = "pytorch"
    label = "PyTorch (torchvision)"
    description = (
        "Image classifiers (ResNet, EfficientNet, MobileNet, ConvNeXt, ViT) fine-tuned from ImageNet weights, and "
        "multilayer perceptrons for tabular classification and regression, spectrogram classifiers for sounds, and "
        "word-level classifiers for texts."
    )
    Hyperparameters = PytorchHyperparameters
    artifacts = ARTIFACTS
    metadata_filename = METADATA_FILENAME
    # A bring-your-own model is saved as a self-contained Hugging Face folder (hf.py), so a trained model loads, serves
    # and exports from the run's results alone, wherever its base model's files were.
    remote_custom_models = True

    @classmethod
    def available(cls) -> str | None:
        # find_spec rather than an import: importing torch is slow and this runs whenever backends are listed.
        for package, name in (("torch", "torch"), ("torchvision", "torchvision"), ("PIL", "pillow")):
            if importlib.util.find_spec(package) is None:
                return f"{name} is not installed"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return (
            task.id in IMAGE_TASKS
            or task.id in TABULAR_TASKS
            or task.id in AUDIO_TASKS
            or task.id in TEXT_TASKS
            or task.id in TOKEN_TASKS
        )

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        if task.id in IMAGE_TASKS:
            return [
                ModelChoice(id=m.id, label=m.label, description=m.description, pretrained=True) for m in IMAGE_MODELS
            ]
        if task.id in TABULAR_TASKS:
            return [ModelChoice(id=m.id, label=m.label, description=m.description) for m in TABULAR_MODELS]
        if task.id in AUDIO_TASKS:
            return [
                ModelChoice(id=m.id, label=m.label, description=m.description, pretrained=m.pretrained)
                for m in AUDIO_MODELS
            ]
        if task.id in TEXT_TASKS:
            return [ModelChoice(id=m.id, label=m.label, description=m.description) for m in TEXT_MODELS]
        if task.id in TOKEN_TASKS:
            return [ModelChoice(id=m.id, label=m.label, description=m.description) for m in TAGGER_MODELS]
        return []

    @classmethod
    def hyperparameter_specs(cls, task: TaskDescriptor) -> list[ParamSpec]:
        return _hyperparameter_specs(task)

    @classmethod
    def custom_model_kinds(cls, task: TaskDescriptor) -> list[CustomModelKind]:
        return _custom.custom_kinds_for(task.id)

    @classmethod
    def suggest_custom_kind(cls, config: dict[str, Any]) -> str | None:
        return _custom.suggest_kind(config)

    @classmethod
    def validate_custom_model(cls, task: TaskDescriptor, ref: CustomModelRef) -> None:
        _custom.validate_custom_model(task, ref)

    @classmethod
    def compile(
        cls,
        task: TaskDescriptor,
        ctx: SnapshotContext,
        hp: PytorchHyperparameters,
        custom: CustomModelRef | None = None,
    ) -> dict[str, Any]:
        return compile_pytorch_config(task, ctx, hp, custom)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from omnitrain.plugins.backends.pytorch.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from omnitrain.plugins.backends.pytorch.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from omnitrain.plugins.backends.pytorch.evaluate import evaluate as _evaluate
        from omnitrain.plugins.backends.pytorch.model import PytorchLoadedModel

        assert isinstance(model, PytorchLoadedModel)
        return _evaluate(model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        from omnitrain.plugins.backends.pytorch.artifacts import convert as _convert
        from omnitrain.plugins.backends.pytorch.model import PytorchLoadedModel

        assert isinstance(model, PytorchLoadedModel)
        return _convert(model, artifact_id, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
