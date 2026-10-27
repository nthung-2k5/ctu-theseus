"""Compile a language-model task plus snapshot context plus hyperparameters into a training config.

The config is a plain dict stored verbatim as `training_runs.config`; `train.py` reads it back. No torch import: this
module is imported when the backend is merely listed.
"""

from typing import Any, Literal

from pydantic import Field

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import (
    ConfigError,
    CustomModelRef,
    HyperparamsBase,
    grad_clip_spec,
    lr_schedule_spec,
    refuse_unsupported,
    seed_spec,
    warmup_spec,
    weight_decay_spec,
)
from omnitrain.plugins.backends.llm import tasks
from omnitrain.plugins.backends.llm.custom import HF_CAUSAL_LM
from omnitrain.schemas.common import ParamSpec

_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_TEXT = "Text"


class LlmHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters)."""

    # LoRA: the rank of the matrices trained beside each frozen weight, how strongly their effect is scaled (the scale
    # is alpha / rank), and the share of their inputs dropped while training.
    lora_rank: int | None = Field(default=None, ge=1, le=256)
    lora_alpha: int | None = Field(default=None, ge=1, le=1024)
    lora_dropout: float | None = Field(default=None, ge=0, lt=1)
    # Text: the most tokens of a prompt and its answer together, and of an answer alone.
    max_length: int | None = Field(default=None, ge=32, le=32768)
    max_new_tokens: int | None = Field(default=None, ge=1, le=8192)
    # Memory: batches added up before one update, recomputing activations instead of keeping them, and holding the
    # frozen base weights in 4 or 8 bits (QLoRA; needs a CUDA GPU and the bitsandbytes package).
    grad_accumulation: int | None = Field(default=None, ge=1, le=512)
    gradient_checkpointing: bool | None = None
    quantization: Literal["none", "4bit", "8bit"] | None = None


def _column(ctx: SnapshotContext, name: str) -> str:
    if not any(c.name == name for c in ctx.columns):
        raise ConfigError(f'The snapshot has no "{name}" column, which this task needs')
    return name


def compile_llm_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: LlmHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    if task.id not in tasks.LLM_TASKS or task.output is None:
        raise ConfigError(f'The language-model backend does not support the task "{task.id}"')
    if custom is None:
        if hp.model_id and hp.model_id.startswith("custom:"):
            raise ConfigError(f'Custom model "{hp.model_id}" was not resolved (it may have been removed or disabled)')
        raise ConfigError(
            "This backend has no built-in models: add a Hugging Face causal language model first, then choose it"
        )
    if custom.kind != HF_CAUSAL_LM:
        raise ConfigError(f'A "{custom.kind}" custom model cannot be used for task "{task.id}"')
    try:
        # The weights are loaded in the precision that suits the device, so a mixed-precision switch has nothing to do.
        refuse_unsupported(hp, "language-model", "mixed_precision")
    except ConfigError as e:
        raise ConfigError(f'{e} for the task "{task.id}"') from None

    d = tasks.DEFAULTS
    max_length = hp.max_length or d["max_length"]
    max_new_tokens = hp.max_new_tokens or d["max_new_tokens"]
    if max_new_tokens >= max_length:
        raise ConfigError(
            f"The longest answer ({max_new_tokens} tokens) must be shorter than the longest prompt and answer together "
            f"({max_length} tokens), or no prompt would be left"
        )
    epochs = hp.epochs or d["epochs"]
    patience = hp.early_stop_patience if hp.early_stop_patience is not None else d["patience"]
    rank = hp.lora_rank or d["lora_rank"]
    return {
        "backend": "llm",
        "task": task.id,
        "model": hp.model_id,
        "custom": {"kind": custom.kind, "path": custom.local_path},
        "inputColumns": [_column(ctx, name) for name in task.input_fields],
        "outputColumn": _column(ctx, task.output.column),
        "text": {"maxLength": max_length, "maxNewTokens": max_new_tokens},
        # Whether the frozen weights are held in 4 or 8 bits is checked when training starts: the machine that trains
        # (a worker, say) is not the one that compiles, and it is the one that has, or lacks, a GPU and bitsandbytes.
        "quantization": hp.quantization or "none",
        "lora": {
            "rank": rank,
            # Twice the rank (a scale of 2) unless asked, so a rank change keeps the update's strength about the same.
            "alpha": hp.lora_alpha or (rank * d["lora_alpha"] // d["lora_rank"]),
            "dropout": d["lora_dropout"] if hp.lora_dropout is None else hp.lora_dropout,
        },
        "train": {
            "epochs": epochs,
            "batchSize": hp.batch_size if isinstance(hp.batch_size, int) else d["batch"],
            "learningRate": hp.learning_rate or d["learning_rate"],
            "optimizer": tasks.OPTIMIZER,
            # -1 switches early stopping off: a patience of `epochs` can never be reached.
            "patience": epochs if patience < 0 else patience,
            "seed": hp.seed,
            "weightDecay": hp.weight_decay,
            "lrSchedule": hp.lr_schedule,
            "warmupEpochs": hp.warmup_epochs,
            "gradClipNorm": hp.grad_clip_norm,
            "mixedPrecision": False,
            "gradAccumulation": hp.grad_accumulation or 1,
            "gradientCheckpointing": bool(hp.gradient_checkpointing),
        },
    }


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    d = tasks.DEFAULTS
    return [
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING, default=d["epochs"], min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch). A "
            "language model usually needs only a few.",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(d["batch"]), choices=[str(n) for n in tasks.BATCH_SIZES],
            description="How many examples the model looks at before it updates itself. A language model is large, "
            "so lower this first if training runs out of memory.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=d["learning_rate"], min=0.000001, max=0.01, step=0.00005,
            description="How big a step the model takes each time it learns from some examples. LoRA trains only a "
            "few small matrices, so it takes a much larger step (around 0.0002) than full fine-tuning would.",
        ),
        seed_spec(_TRAINING),
        weight_decay_spec(_TRAINING),
        lr_schedule_spec(_TRAINING),
        warmup_spec(_TRAINING),
        grad_clip_spec(_TRAINING),
        ParamSpec(
            name="earlyStopPatience", label="Early Stop Patience", type="int", group=_STOPPING,
            default=d["patience"], min=1, step=1, disabled_value=-1,
            description="Stops training when validation results have not improved for this many epochs in a row.",
        ),
        ParamSpec(
            name="loraRank", label="LoRA rank", type="choice", group=_MODEL,
            default=str(d["lora_rank"]), choices=[str(n) for n in tasks.LORA_RANKS],
            description="How much the model can change: the size of the small matrices trained beside each frozen "
            "weight. A higher rank can learn more from many examples, and costs more memory.",
        ),
        ParamSpec(
            name="loraAlpha", label="LoRA alpha", type="int", group=_MODEL, default=None, min=1, max=1024, step=1,
            description="How strongly the trained matrices are applied (scaled by alpha divided by the rank). Leave "
            "it unset to use twice the rank.",
        ),
        ParamSpec(
            name="loraDropout", label="LoRA dropout", type="float", group=_MODEL,
            default=d["lora_dropout"], min=0, max=0.9, step=0.05,
            description="Randomly drops this share of the values the trained matrices see while training, so they "
            "cannot lean on any one of them. Raise it if the model memorises a small training set.",
        ),
        ParamSpec(
            name="gradAccumulation", label="Batches per update", type="int", group=_TRAINING,
            default=1, min=1, max=512, step=1,
            description="Adds up the learning from this many batches before the model updates itself, which trains "
            "as if the batch were that many times bigger without needing the memory for it. Use it when the batch "
            "size you want does not fit on the GPU.",
        ),
        ParamSpec(
            name="gradientCheckpointing", label="Save memory (checkpointing)", type="bool", group=_TRAINING,
            default=False,
            description="Recomputes part of the work instead of keeping it in memory. Training is about a third "
            "slower, and a longer prompt or a bigger model fits on the GPU.",
        ),
        ParamSpec(
            name="quantization", label="Hold the base model in fewer bits", type="choice", group=_MODEL,
            default="none", choices=["none", "4bit", "8bit"],
            description="Stores the frozen base model in 4 or 8 bits instead of 16 or 32 (QLoRA), so a model several "
            "times larger fits on the GPU, at a small cost in quality and speed. Needs a CUDA GPU and the bitsandbytes "
            "package. The trained model is saved at full precision either way.",
        ),
        ParamSpec(
            name="maxLength", label="Longest prompt and answer (tokens)", type="int", group=_TEXT,
            default=d["max_length"], min=32, max=32768, step=32,
            description="The most tokens of a prompt and its answer together. A longer example loses the start of its "
            "prompt, never its answer. Memory grows quickly with this number, and the model cannot read more tokens "
            "than its own limit.",
        ),
        ParamSpec(
            name="maxNewTokens", label="Longest answer (tokens)", type="int", group=_TEXT,
            default=d["max_new_tokens"], min=1, max=8192, step=1,
            description="The most tokens of an answer: training cuts a longer one, and the model stops writing here "
            "when it is asked. Must be shorter than the longest prompt and answer together.",
        ),
    ]  # fmt: skip
