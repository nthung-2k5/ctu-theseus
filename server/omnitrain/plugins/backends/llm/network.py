"""A Hugging Face causal language model, with LoRA adapters while it trains. Imports torch (and transformers / peft
inside functions): only loaded where a model is really built.

Nothing here ever trusts remote code or unpickles weights: local files only, safetensors only.

Training wraps the model in `CausalNet`, so the generic loop in `pytorch/loop.py` can drive it: one packed input tensor
in, the next-token logits out. The adapters are merged into the weights when training ends (`merge`), so the saved
model is an ordinary Hugging Face folder that loads, serves and evaluates with no `peft` and no base model next to it.
"""

import importlib.util
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

# The LoRA matrices are the only weights that change; the loop keeps and restores only these for the best epoch.
ADAPTER_MARK = "lora_"


def is_adapter(name: str) -> bool:
    return ADAPTER_MARK in name


def _options() -> dict[str, object]:
    return {"local_files_only": True, "trust_remote_code": False}


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, **_options())
    if tokenizer.eos_token_id is None:
        raise ValueError("its tokenizer has no end-of-text token, which is how a written answer is known to be over")
    if not tokenizer("a short text", add_special_tokens=False)["input_ids"]:
        raise ValueError("its tokenizer reads a text as no tokens at all (an empty vocabulary?)")
    return tokenizer


def pad_id(tokenizer: Any) -> int:
    """The id padding is written with. The attention mask says what is padding, never the id, so a model with no pad
    token can borrow the end-of-text token."""
    return int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)


QUANTIZATIONS = ("none", "4bit", "8bit")


def quantization_unavailable(quantization: str) -> str | None:
    """Why the base model cannot be held in `quantization`, or None when it can (or is not asked to be)."""
    if quantization == "none":
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        return "the bitsandbytes package is not installed"
    if not torch.cuda.is_available():
        return "it needs a CUDA GPU"
    return None


def _quantization_config(quantization: str, compute_dtype: torch.dtype) -> Any:
    from transformers import BitsAndBytesConfig

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    # NF4 with a second round of quantisation for the scales: the QLoRA recipe.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base(path: str, dtype: torch.dtype, quantization: str = "none") -> nn.Module:
    """The model at `path`, with every tensor read from its files. transformers only warns about a tensor it cannot
    find and makes one up, which would train and serve a model with random layers under a pretrained name.

    `quantization` "4bit" or "8bit" holds the weights of the linear layers in that many bits, on the GPU, for training
    adapters on a model that would not otherwise fit. Such a model is for training only: it cannot be merged into."""
    from transformers import AutoModelForCausalLM

    extra: dict[str, Any] = {}
    if quantization != "none":
        extra = {"quantization_config": _quantization_config(quantization, dtype), "device_map": {"": 0}}
    model, info = AutoModelForCausalLM.from_pretrained(
        path, output_loading_info=True, use_safetensors=True, dtype=dtype, **extra, **_options()
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        missing = sorted(info.get("missing_keys") or [])[:3]
        raise ValueError(f"the model's files are incomplete (missing {', '.join(missing) or 'matching weights'})")
    return model


def smoke_test(path: str) -> None:
    """Read the tokenizer and construct the network from its config, without reading the weights. Raises if either
    fails."""
    from transformers import AutoConfig, AutoModelForCausalLM

    load_tokenizer(path)
    config = AutoConfig.from_pretrained(path, **_options())
    with torch.device("meta"):
        AutoModelForCausalLM.from_config(config, trust_remote_code=False)


class CausalNet(nn.Module):
    """`[batch, 2, length]` ids and attention mask in, `[batch, length - 1, vocabulary]` logits out: the logits at
    position t score token t + 1, so the last position (nothing follows it) is dropped and the targets are the labels
    shifted left by one. The mask travels with the ids because padding has no id of its own to recognise it by."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=packed[:, 0], attention_mask=packed[:, 1], use_cache=False)
        return output.logits[:, :-1]


def prepare(model: nn.Module, *, quantized: bool, checkpointing: bool) -> nn.Module:
    """`model` ready to have adapters trained on it. A quantised model needs its normalisation layers in full precision
    (`prepare_model_for_kbit_training`, which also turns checkpointing on when asked). Checkpointing recomputes the
    activations in the backward pass; the embeddings are frozen, so their output must be told to carry gradients or the
    recomputed segments would have none to pass on."""
    if quantized:
        from peft import prepare_model_for_kbit_training

        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # type: ignore[operator]
        model.enable_input_require_grads()  # type: ignore[operator]
    return model


def add_lora(model: nn.Module, rank: int, alpha: int, dropout: float) -> nn.Module:
    """`model` with a LoRA adapter on every linear layer except the output head. The base weights are frozen."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"
    )
    return get_peft_model(model, config)


def merge(peft_model: nn.Module) -> nn.Module:
    """The plain model with the adapters folded into its weights (the adapters are gone afterwards)."""
    return peft_model.merge_and_unload()  # type: ignore[operator]


def merge_saved(path: str, adapter_dir: str, dtype: torch.dtype) -> nn.Module:
    """The model at `path`, loaded at full precision, with the adapters saved in `adapter_dir` folded in. How a model
    trained on a quantised base is merged: weights held in 4 bits cannot absorb a correction, so the adapters (which
    do not depend on how the base was stored) go onto a fresh full-precision copy."""
    from peft import PeftModel

    return PeftModel.from_pretrained(load_base(path, dtype), adapter_dir).merge_and_unload()  # type: ignore[operator]


def save_adapter(peft_model: nn.Module) -> "tempfile.TemporaryDirectory[str]":
    """The adapters written to a temporary folder (the caller keeps it alive until they have been read back)."""
    folder = tempfile.TemporaryDirectory()
    peft_model.save_pretrained(folder.name)  # type: ignore[operator]
    return folder


def save(model: nn.Module, tokenizer: Any, dest: str) -> None:
    """A self-contained folder `load_base` and `load_tokenizer` read with no other file."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=True)  # type: ignore[operator]
    tokenizer.save_pretrained(dest)
