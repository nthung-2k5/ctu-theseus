"""Checks on a bring-your-own model's files, run by the validation job after its generic checks.

No ludwig/torch import: it only reads JSON, so validating a model never needs a GPU runtime.
`timm` is the exception, imported inside its own branch because that kind has no files to read.
"""

import json
import re
from pathlib import Path

from theseus.backends.base import ConfigError, CustomModelRef
from theseus.backends.ludwig.tasks import HF_CAUSAL_LM, HF_TRANSFORMER, TIMM_IMAGE, custom_kinds_for
from theseus.services.task_registry import TaskDescriptor

# A decoder-only model's `architectures` entry, e.g. LlamaForCausalLM or GPT2LMHeadModel.
_CAUSAL_LM = re.compile(r"(ForCausalLM|LMHeadModel)$")


def validate_custom_model(task: TaskDescriptor, ref: CustomModelRef) -> None:
    if ref.kind not in {k.id for k in custom_kinds_for(task.id)}:
        raise ConfigError(f'A "{ref.kind}" model cannot be used for {task.label}')
    if ref.kind == TIMM_IMAGE:
        _validate_timm(ref)
    elif ref.kind in (HF_TRANSFORMER, HF_CAUSAL_LM):
        _validate_hf(ref)
    else:
        raise ConfigError(f'Unknown custom model kind "{ref.kind}"')


def _validate_timm(ref: CustomModelRef) -> None:
    if not ref.source_ref:
        raise ConfigError("A timm model needs its timm model name (e.g. resnet50.a1_in1k)")
    try:
        import timm
    except ImportError:
        raise ConfigError("The 'timm' package is not installed on this server") from None
    # A timm name is "<architecture>[.<pretrained tag>]"; the architecture is what must exist.
    architecture = ref.source_ref.split(".", 1)[0]
    if not timm.is_model(architecture):
        raise ConfigError(f'"{architecture}" is not a timm architecture')


def _validate_hf(ref: CustomModelRef) -> None:
    root = Path(ref.local_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ConfigError("The model has no config.json at its top level (expected a Hugging Face model folder)")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"config.json could not be read: {e}") from None
    if not isinstance(config, dict) or "model_type" not in config:
        raise ConfigError("config.json has no model_type, so this is not a Hugging Face transformers model")
    if "auto_map" in config:
        # auto_map makes transformers import and run Python from the model repo. Never do that with
        # a file a user (or a Hub author) supplied.
        raise ConfigError("Models that need custom code (auto_map / trust_remote_code) are not supported")
    if not any(root.glob("*.safetensors")):
        raise ConfigError("The model has no .safetensors weights (pickle-based weights are not accepted)")

    if ref.kind == HF_CAUSAL_LM:
        architectures = config.get("architectures") or []
        if not any(_CAUSAL_LM.search(a) for a in architectures):
            listed = ", ".join(architectures) or "none listed"
            raise ConfigError(f"Not a causal language model (architectures: {listed})")
