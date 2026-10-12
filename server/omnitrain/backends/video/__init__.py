"""The video trainer backend (`id = "video"`): fine-tunes torchvision's Kinetics-400 video classifiers.

Submodules:
  tasks.py      the model catalog and clip-sampling defaults (pure data)
  custom.py     bring-your-own Hugging Face video models: kinds, file checks, the clip a model needs
  hf.py         builds, saves and loads those models as a network (torch, transformers)
  compile.py    task + snapshot + hyperparameters -> training config dict (no torch import)
  runtime.py    where weights are cached, device and dataloader choice (torch imported only inside functions)
  frames.py     frame selection, decoding with OpenCV, resize/crop and normalisation (shared by train, serve, docs)
  data.py       snapshot parquet -> local clip files and labelled samples per split (pandas only)
  network.py    builds a torchvision model with a fresh head
  train.py      the training loop, progress/abort, uploading the result
  model.py      VideoLoadedModel: predict/to_output/golden_prediction
  evaluate.py   the evaluation report (the shared classification report)
  artifacts.py  ONNX conversion
  manifest.py   preprocessing.json for a bare model export

Only tasks.py, compile.py, runtime.py, manifest.py and artifacts.py are imported at module level here, and none of
them imports torch at module level, so listing this backend never needs a GPU runtime. The others import torch, so
they are imported lazily, inside the classmethods that need them.

Licence: torchvision and its pretrained Kinetics weights are BSD-3-Clause.
"""

import importlib.util
from typing import Any

import pandas as pd

from omnitrain.backends.base import (
    CustomModelKind,
    CustomModelRef,
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from omnitrain.backends.video import custom as _custom
from omnitrain.backends.video.artifacts import ARTIFACTS
from omnitrain.backends.video.compile import VideoHyperparameters, compile_video_config
from omnitrain.backends.video.compile import hyperparameter_specs as _hyperparameter_specs
from omnitrain.backends.video.manifest import METADATA_FILENAME
from omnitrain.backends.video.tasks import VIDEO_MODELS, VIDEO_TASKS
from omnitrain.schemas.common import ParamSpec
from omnitrain.services.task_registry import SnapshotContext, TaskDescriptor


class VideoBackend(TrainerBackend):
    id = "video"
    label = "Video (torchvision)"
    description = "3D-convolutional and transformer video classifiers, fine-tuned from Kinetics-400 weights."
    Hyperparameters = VideoHyperparameters
    artifacts = ARTIFACTS
    metadata_filename = METADATA_FILENAME
    # A fine-tuned Hugging Face video model is saved as a self-contained folder (backends/video/hf.py), so it never
    # needs the original files again and the run may train on a worker.
    remote_custom_models = True

    @classmethod
    def available(cls) -> str | None:
        # find_spec rather than an import: importing torch/cv2 is slow and this runs whenever backends are listed.
        for package, name in (("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv-python")):
            if importlib.util.find_spec(package) is None:
                return f"{name} is not installed"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id in VIDEO_TASKS

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
        if task.id not in VIDEO_TASKS:
            return []
        return [ModelChoice(id=m.id, label=m.label, description=m.description, pretrained=True) for m in VIDEO_MODELS]

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
        hp: VideoHyperparameters,
        custom: CustomModelRef | None = None,
    ) -> dict[str, Any]:
        return compile_video_config(task, ctx, hp, custom)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from omnitrain.backends.video.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from omnitrain.backends.video.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from omnitrain.backends.video.evaluate import evaluate as _evaluate
        from omnitrain.backends.video.model import VideoLoadedModel

        assert isinstance(model, VideoLoadedModel)
        return _evaluate(model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        from omnitrain.backends.video.artifacts import convert as _convert
        from omnitrain.backends.video.model import VideoLoadedModel

        assert isinstance(model, VideoLoadedModel)
        return _convert(model, artifact_id, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from omnitrain.backends.video.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
