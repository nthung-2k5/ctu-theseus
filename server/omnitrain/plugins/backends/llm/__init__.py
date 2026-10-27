"""The language-model trainer backend (`id = "llm"`): LoRA fine-tuning of a bring-your-own causal language model.

Free text in, free text out: text generation, summarization, sequence to sequence and question answering. A prompt is
the task's input fields under headings (`prompt.py`), the answer is what the model writes next, and training grades only
the answer's tokens. Only the small LoRA matrices are trained; when training ends they are merged into the weights and
the whole model is saved as a Hugging Face folder, so a trained run loads and serves from its results alone and can
train on a remote worker.

There are no built-in models (a language model is too big to ship): every run uses one added through the "add a model"
flow (a Hub reference or an upload), validated in `custom.py`. The one export is the merged folder itself
(`artifacts.py`, the `hf_model` format): a generative model has no fixed-shape ONNX graph the generated clients could
run, so the generated devkits and apps do not apply.

Submodules:
  tasks.py     the tasks and the defaults (pure data)
  prompt.py    the one prompt layout (pure)
  metrics.py   exact match, token F1 and ROUGE between a written and a generated text (pure)
  custom.py    the bring-your-own kind and its file checks (JSON only, no torch)
  compile.py   task + snapshot + hyperparameters -> training config dict (no torch import)
  runtime.py   device and number format (torch imported only inside functions)
  data.py      snapshot rows -> padded token batches (torch)
  network.py   loads the model, adds and merges the LoRA adapters (torch, transformers, peft)
  train.py     the flow: fine-tune, merge, upload
  model.py     LlmLoadedModel: predict (generate) / to_output
  evaluate.py  the text-score report
  artifacts.py the export: the merged folder, zipped (torch)
  manifest.py  the export's preprocessing.json (pure)

Only tasks.py, custom.py, compile.py and runtime.py are imported at module level here, and none of them imports torch at
module level, so listing this backend never needs a GPU runtime.

A model that does not fit in memory can be trained on a 4-bit / 8-bit base (QLoRA), with gradients added up over
several batches and with gradient checkpointing (`compile.py`, `network.py`).

Not offered yet: saving only the adapter, sampling settings for generation (it is always the most likely
continuation).
"""

import importlib.util
from typing import Any

import pandas as pd

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import (
    ARTIFACT_FILENAMES,
    Artifact,
    CustomModelKind,
    CustomModelRef,
    EvalResult,
    LoadedModel,
    ModelChoice,
    TrainContext,
    TrainerBackend,
)
from omnitrain.plugins.backends.llm import custom as _custom
from omnitrain.plugins.backends.llm.compile import LlmHyperparameters, compile_llm_config
from omnitrain.plugins.backends.llm.compile import hyperparameter_specs as _hyperparameter_specs
from omnitrain.plugins.backends.llm.runtime import METADATA_FILENAME
from omnitrain.plugins.backends.llm.tasks import LLM_TASKS
from omnitrain.schemas.common import ParamSpec


class LlmBackend(TrainerBackend):
    id = "llm"
    label = "Language model (LoRA)"
    description = (
        "Fine-tunes a Hugging Face causal language model with LoRA adapters to write text: generation, summarization, "
        "sequence to sequence and question answering. Bring your own base model."
    )
    Hyperparameters = LlmHyperparameters
    metadata_filename = METADATA_FILENAME
    # The adapters are merged into a copy of the model, saved as a Hugging Face folder that holds everything, so a
    # trained run loads and serves from its results alone, wherever its base model's files were.
    remote_custom_models = True
    # The merged Hugging Face folder, zipped: what the `hf_model` export format bundles.
    artifacts = {"hf_model": Artifact("hf_model", ARTIFACT_FILENAMES["hf_model"])}

    @classmethod
    def available(cls) -> str | None:
        # find_spec rather than an import: importing torch is slow and this runs whenever backends are listed.
        for package in ("torch", "transformers", "peft"):
            if importlib.util.find_spec(package) is None:
                return f"{package} is not installed"
        return None

    @classmethod
    def supports(cls, task: TaskDescriptor) -> bool:
        return task.id in LLM_TASKS

    @classmethod
    def models(cls, task: TaskDescriptor) -> list[ModelChoice]:
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
        hp: LlmHyperparameters,
        custom: CustomModelRef | None = None,
    ) -> dict[str, Any]:
        return compile_llm_config(task, ctx, hp, custom)

    @classmethod
    def train(cls, run: TrainContext) -> LoadedModel:
        from omnitrain.plugins.backends.llm.train import train as _train

        return _train(run)

    @classmethod
    def load(cls, model_dir: str) -> LoadedModel:
        from omnitrain.plugins.backends.llm.model import load as _load

        return _load(model_dir)

    @classmethod
    def evaluate(
        cls, model: LoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str
    ) -> EvalResult | None:
        from omnitrain.plugins.backends.llm.evaluate import evaluate as _evaluate
        from omnitrain.plugins.backends.llm.model import LlmLoadedModel

        assert isinstance(model, LlmLoadedModel)
        return _evaluate(model, df, split_column, item_id_column)

    @classmethod
    def convert(cls, model: LoadedModel, artifact_id: str, workdir: str) -> str:
        if artifact_id != "hf_model":
            raise ValueError(f"The language-model backend produces no {artifact_id!r} artifact")
        from omnitrain.plugins.backends.llm.artifacts import export_hf_model
        from omnitrain.plugins.backends.llm.model import LlmLoadedModel

        assert isinstance(model, LlmLoadedModel)
        return export_hf_model(model, workdir)

    @classmethod
    def preprocessing_manifest(cls, run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
        from omnitrain.plugins.backends.llm.manifest import build_preprocessing_manifest

        return build_preprocessing_manifest(run_id, config, meta)
