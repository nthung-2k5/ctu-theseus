"""Bring-your-own causal language models (Hugging Face `transformers`: Llama, Qwen, Mistral, Phi, Gemma, GPT-2, ...).

This backend has no built-in models: a language model is far too big to ship, so every run starts from one the user (or
an admin) added. This module only reads JSON, so listing the kind and checking a model's files needs no torch; building
the network is `network.py`.

What a model must be: a decoder-only checkpoint (`...ForCausalLM`, or GPT-2's `...LMHeadModel`) with `.safetensors`
weights, a tokenizer and an end-of-text token (the model learns to write it, which is how serving knows an answer is
over). Code that ships with a model (`auto_map`) is never run.
"""

import importlib.util
import re
from pathlib import Path
from typing import Any

from omnitrain.domain.task_registry import TaskDescriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelKind, CustomModelRef
from omnitrain.plugins.backends.llm import tasks
from omnitrain.plugins.backends.pytorch.custom import read_config

HF_CAUSAL_LM = "hf_causal_lm"
_ARCHITECTURE = re.compile(r"(ForCausalLM|LMHeadModel)$")
# What a published tokenizer is made of (the fast tokenizer's file, or the vocabulary files it is converted from).
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "spiece.model")
# The config keys that say how many positions a model reads, by the name each family gave them.
_POSITION_KEYS = ("max_position_embeddings", "n_positions", "n_ctx", "seq_length")


def _unavailable() -> str | None:
    for package in ("transformers", "peft"):
        if importlib.util.find_spec(package) is None:
            return f"The '{package}' package is not installed on this server"
    return None


def custom_kinds_for(task_id: str) -> list[CustomModelKind]:
    if task_id not in tasks.LLM_TASKS:
        return []
    return [
        CustomModelKind(
            id=HF_CAUSAL_LM,
            label="Hugging Face causal language model",
            description="A decoder-only language model (Llama, Qwen, Mistral, Phi, Gemma, GPT-2, ...) as config.json, "
            ".safetensors weights and tokenizer files. It is fine-tuned with LoRA adapters, which are merged into a "
            "copy of the model when training ends.",
            modality="text",
            source_kinds=["hub", "upload"],
            status="experimental",
            unavailable_reason=_unavailable(),
        )
    ]


def suggest_kind(config: dict[str, Any]) -> str | None:
    architectures = [a for a in config.get("architectures") or [] if isinstance(a, str)]
    return HF_CAUSAL_LM if any(_ARCHITECTURE.search(a) for a in architectures) else None


def position_limit(config: dict[str, Any]) -> int | None:
    """The most tokens the model reads at once, if its config says."""
    for key in _POSITION_KEYS:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def check_files(root: Path) -> dict[str, Any]:
    """The config of the model at `root`, after the checks that need no torch. Raises `ConfigError`."""
    config = read_config(root)
    if "auto_map" in config:
        # auto_map makes transformers import and run Python from the model repo. Never do that with a file a user (or
        # a Hub author) supplied.
        raise ConfigError("Models that need custom code (auto_map / trust_remote_code) are not supported")
    architectures = [a for a in config.get("architectures") or [] if isinstance(a, str)]
    if not any(_ARCHITECTURE.search(a) for a in architectures):
        raise ConfigError(
            f"Not a causal language model (architectures {architectures or 'missing'}; expected one ending in "
            "ForCausalLM or LMHeadModel)"
        )
    if not any(root.glob("*.safetensors")):
        raise ConfigError("The model has no .safetensors weights (pickle-based weights are not accepted)")
    if not any((root / name).is_file() for name in TOKENIZER_FILES):
        raise ConfigError("The model has no tokenizer files (expected tokenizer.json or a vocabulary file)")
    return config


def validate_custom_model(task: TaskDescriptor, ref: CustomModelRef) -> None:
    if ref.kind not in {k.id for k in custom_kinds_for(task.id)}:
        raise ConfigError(f'A "{ref.kind}" model cannot be used for {task.label}')
    root = Path(ref.local_path)
    check_files(root)

    # Prove it works without reading the weights (a 7B model would take minutes and tens of GB): the network must be
    # constructible from its config, and the tokenizer must read a text and know where a text ends. The weights
    # themselves are checked strictly when training starts (`network.load_base`).
    try:
        from omnitrain.plugins.backends.llm import network

        network.smoke_test(str(root))
    except Exception as e:  # noqa: BLE001  any failure here means the model cannot be fine-tuned as a language model
        reason = str(e).strip() or type(e).__name__
        raise ConfigError(f"This model cannot be loaded as a language model: {reason.splitlines()[0][:300]}") from None
