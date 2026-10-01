"""Compile a task descriptor plus snapshot context plus hyperparameters into a Ludwig config.

Ported from server/lib/ludwig/{compile,schema,serialize}.ts, then from services/ludwig_config.py
when trainer backends became a plugin system. The compiled config is validated here, before a run
row is ever committed, so an invalid configuration surfaces as a 400 rather than as a worker crash
discovered later.
"""

import copy
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from theseus import constants as C
from theseus.backends.base import ConfigError, CustomModelRef, HyperparamsBase
from theseus.backends.ludwig.tasks import (
    HF_CAUSAL_LM,
    HF_TRANSFORMER,
    HF_VISION,
    LUDWIG_OPTIMIZER_TYPES,
    LUDWIG_TASKS,
    TIMM_IMAGE,
    EncoderChoice,
    custom_kinds_for,
)
from theseus.schemas.common import ParamSpec
from theseus.services.task_registry import SnapshotContext, TaskDescriptor

LUDWIG_VERSION = "0.17.5"

# Curated choice menus. Ludwig itself doesn't validate these (validation_metric is passed straight
# through and rejected by Ludwig only if the output type doesn't support it), so they live here as
# a UI convenience rather than as pydantic Literal fields on LudwigHyperparameters.
_CLASSIFICATION_METRICS = ["loss", "accuracy"]
_REGRESSION_METRICS = ["loss", "mean_squared_error", "mean_absolute_error", "r2"]
_IMAGE_SIZES = ["128", "224", "256"]

# Section headings of the create-run form (ParamSpec.group), in the order they appear.
# Section headings of the hyperparameter form, in the order it shows them: what the model is, how it trains,
# when it stops, what data it sees. In plain words, since the people filling the form may not know the ML terms.
_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_DATA = "Data"


class LudwigHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `model_id` (from HyperparamsBase) is the encoder id from the task's encoder catalog; defaults
    to the first. Kept as `encoderId` on the wire (the field predates the generic `model_id` name
    and every existing client already sends it) rather than picking up HyperparamsBase's generated
    `modelId` alias. There is deliberately no preprocessing or augmentation here: both moved to
    snapshot creation, where the resulting items are real, browsable dataset items (see
    services/preprocessing.py and services/augmentation.py).
    """

    model_id: str | None = Field(default=None, alias="encoderId")
    # Weight each class loss inversely to its snapshot frequency (category outputs only).
    use_class_weights: bool | None = None
    # Square-resize every image input feature to this many pixels per side. Ignored for non-vision tasks.
    image_size: int | None = None
    # Metric tracked for early stopping / best epoch instead of Ludwig's per-output-type default.
    # Passed straight through: Ludwig itself rejects a name the output feature type does not support.
    validation_metric: str | None = None
    # Defaults to Ludwig's per-model-type default (Adam for ECD) when unset.
    optimizer: str | None = None
    # Train only the head: the encoder's pretrained weights stay fixed. Pretrained encoders only.
    freeze_backbone: bool | None = None
    # The head is the combiner's fully-connected stack (ECD tasks). 0 layers is a plain linear head.
    head_layers: int | None = Field(default=None, ge=0, le=4)
    head_width: int | None = Field(default=None, ge=8, le=2048)
    head_dropout: float | None = Field(default=None, ge=0, le=0.9)
    # Truncate text/sequence inputs to this many tokens. Ignored for other input types.
    max_sequence_length: int | None = Field(default=None, ge=8, le=4096)


# -- Validation schema (replaces the zod schema) ---------------------------------------------


class _Feature(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: Literal["image", "text", "audio", "number", "category", "binary", "sequence", "vector"]
    column: str


class _Typed(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class _Combiner(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class _InputFeature(_Feature):
    encoder: _Typed | None = None


class _OutputFeature(_Feature):
    loss: _Typed | None = None


class _Trainer(BaseModel):
    epochs: int = Field(gt=0)
    batch_size: int | Literal["auto"]
    learning_rate: float = Field(gt=0)
    early_stop: int
    # Declared explicitly so a strict model never silently strips them.
    validation_metric: str | None = None
    optimizer: _Typed | None = None


class _FixedSplit(BaseModel):
    type: Literal["fixed"]
    column: str


class _Preprocessing(BaseModel):
    split: _FixedSplit


class LudwigConfig(BaseModel):
    model_type: Literal["ecd", "llm"]
    # The LLM base model. Declared so `model_dump` keeps it: unknown keys are dropped, and a missing
    # base_model makes Ludwig reject an llm config.
    base_model: str | None = None
    input_features: list[_InputFeature] = Field(min_length=1)
    output_features: list[_OutputFeature] = Field(min_length=1)
    combiner: _Combiner | None = None
    preprocessing: _Preprocessing
    trainer: _Trainer
    ludwig_version: str


# -- Compiler --------------------------------------------------------------------------------


def compile_ludwig_config(
    task: TaskDescriptor,
    ctx: SnapshotContext,
    hp: LudwigHyperparameters | None = None,
    custom: CustomModelRef | None = None,
) -> dict[str, Any]:
    sel = hp or LudwigHyperparameters()
    spec = LUDWIG_TASKS.get(task.id)
    if spec is None:
        raise ConfigError(f'Task "{task.id}" has no Ludwig backend (status: {task.status})')
    knobs = spec.trainer_knobs

    if custom is None and sel.model_id and sel.model_id.startswith(CUSTOM_MODEL_PREFIX):
        raise ConfigError(f'Custom model "{sel.model_id}" was not resolved (it may have been removed or disabled)')
    if custom is not None:
        _check_custom_kind(task.id, custom)
    if sel.freeze_backbone:
        if custom is not None and custom.kind == HF_CAUSAL_LM:
            # A language model is fine-tuned whole; there is no separate encoder whose weights could be held fixed.
            raise ConfigError("Freezing the backbone does not apply to a language model base model")
        # A custom encoder is always pretrained, so it can be frozen wherever the built-in catalog could not.
        if custom is None and not any(e.pretrained for e in spec.encoders):
            raise ConfigError(f'Task "{task.id}" has no pretrained backbone to freeze')
    combiner = _head_combiner(sel, task.id, spec.model_type)

    declared = copy.deepcopy(spec.input_features)
    if declared:
        input_features = [
            _with_sequence_length(
                _with_image_resize(
                    _with_encoder(f, spec.encoders, sel.model_id, sel.freeze_backbone, custom), sel.image_size
                ),
                sel.max_sequence_length,
            )
            for f in declared
        ]
    else:
        # Tabular tasks do not know their column names statically: derive one number feature
        # per scalar column the snapshot actually contains.
        input_features = [
            {"name": c.name, "type": "number", "column": c.name} for c in ctx.columns if c.kind == "scalar"
        ]

    if not input_features:
        raise ConfigError(f'Task "{task.id}": no input features could be derived from the snapshot columns')

    output_features = copy.deepcopy(spec.output_features)
    if sel.use_class_weights:
        output_features = _apply_class_weights(output_features, ctx.class_counts, task.id)

    trainer: dict[str, Any] = {
        "epochs": sel.epochs if sel.epochs is not None else knobs.epochs.default,
        "batch_size": sel.batch_size if sel.batch_size is not None else knobs.batch_size.default,
        "learning_rate": sel.learning_rate if sel.learning_rate is not None else knobs.learning_rate.default,
        "early_stop": sel.early_stop_patience
        if sel.early_stop_patience is not None
        else knobs.early_stop_patience.default,
    }
    if sel.validation_metric:
        trainer["validation_metric"] = sel.validation_metric
    if sel.optimizer:
        trainer["optimizer"] = _with_optimizer(sel.optimizer, task.id)

    config = {
        "model_type": spec.model_type,
        # An LLM task fine-tunes its base model directly: it has no encoder to swap, so the custom
        # model is the base_model. (An LLM task offers no built-in models, so without one it stays unset.)
        **({"base_model": custom.local_path} if custom is not None and custom.kind == HF_CAUSAL_LM else {}),
        "input_features": input_features,
        "output_features": output_features,
        **({"combiner": combiner} if combiner else {}),
        # Without this Ludwig re-splits randomly 70/10/20 and ignores the split the user
        # assigned. It must be the synthetic integer column, not the human-readable one
        # (see services/snapshot.py).
        "preprocessing": {"split": {"type": "fixed", "column": C.SPLIT_INDEX_COLUMN_NAME}},
        "trainer": trainer,
        "ludwig_version": LUDWIG_VERSION,
    }
    return LudwigConfig.model_validate(config).model_dump(exclude_none=True)


def serialize_ludwig_config(config: dict[str, Any]) -> str:
    """Serialize a compiled Ludwig config to YAML for config.yaml."""
    return yaml.safe_dump(config, sort_keys=False)


# Every custom model's id in `hyperparameters.encoderId`, e.g. "custom:0192...". See services/model_catalog.py.
CUSTOM_MODEL_PREFIX = "custom:"


def _check_custom_kind(task_id: str, custom: CustomModelRef) -> None:
    if custom.kind not in {k.id for k in custom_kinds_for(task_id)}:
        raise ConfigError(f'A "{custom.kind}" custom model cannot be used for task "{task_id}"')


def _custom_encoder(custom: CustomModelRef, freeze: bool | None) -> dict[str, Any] | None:
    """The Ludwig encoder block for a custom model, or None when the kind is not an encoder swap."""
    trainable = {"trainable": False} if freeze else {}
    if custom.kind == HF_TRANSFORMER:
        # auto_transformer always loads pretrained weights, from a local directory here.
        return {"type": "auto_transformer", "pretrained_model_name_or_path": custom.local_path, **trainable}
    if custom.kind == HF_VISION:
        # This backend's own encoder (backends/ludwig/encoders.py), which loads any transformers vision model.
        return {"type": "hf_vision", "pretrained_model_name_or_path": custom.local_path, **trainable}
    if custom.kind == TIMM_IMAGE:
        return {"type": "timm", "model_name": custom.source_ref, "use_pretrained": True, **trainable}
    return None


def _with_encoder(
    feature: dict[str, Any],
    encoders: list[EncoderChoice],
    model_id: str | None,
    freeze: bool | None = None,
    custom: CustomModelRef | None = None,
) -> dict[str, Any]:
    if custom is not None:
        # Bypass the built-in catalog entirely: the model id is "custom:{uuid}", not one of its entries.
        encoder = _custom_encoder(custom, freeze)
        return feature if encoder is None or feature.get("encoder") else {**feature, "encoder": encoder}
    if feature.get("encoder") or not encoders:
        return feature
    if model_id:
        encoder = next((e for e in encoders if e.id == model_id), None)
    else:
        encoder = encoders[0]
    if encoder is None:
        available = ", ".join(e.id for e in encoders)
        raise ConfigError(f'Unknown encoder "{model_id}" (available: {available})')
    if freeze and not encoder.pretrained:
        raise ConfigError(f'"{encoder.label}" is trained from scratch, so there are no pretrained weights to freeze')
    return {
        **feature,
        "encoder": {
            "type": encoder.encoder_type,
            "use_pretrained": encoder.pretrained,
            **(encoder.params or {}),
            **({"trainable": False} if freeze else {}),
        },
    }


def _with_image_resize(feature: dict[str, Any], size: int | None) -> dict[str, Any]:
    """Square-resize an image input feature. A no-op for every non-image feature or unset size."""
    if not size or feature["type"] != "image":
        return feature
    return {**feature, "preprocessing": {**feature.get("preprocessing", {}), "height": size, "width": size}}


def _with_sequence_length(feature: dict[str, Any], length: int | None) -> dict[str, Any]:
    """Truncate a text/sequence input feature to `length` tokens. A no-op for every other feature type."""
    if not length or feature["type"] not in ("text", "sequence"):
        return feature
    return {**feature, "preprocessing": {**feature.get("preprocessing", {}), "max_sequence_length": length}}


def _head_combiner(sel: LudwigHyperparameters, task_id: str, model_type: str) -> dict[str, Any] | None:
    """The combiner section for the requested head, or None when the head was left at Ludwig's default.

    The combiner joins every input feature's encoder output and feeds the output decoders; its
    fully-connected stack is what the UI calls the head (0 layers = a plain linear head).
    """
    requested = {
        "num_fc_layers": sel.head_layers,
        "output_size": sel.head_width,
        "dropout": sel.head_dropout,
    }
    requested = {k: v for k, v in requested.items() if v is not None}
    if not requested:
        return None
    if model_type != "ecd":
        raise ConfigError(f'Task "{task_id}" has no configurable head (it fine-tunes a language model directly)')
    return {"type": "concat", **requested}


def _with_optimizer(optimizer: str, task_id: str) -> dict[str, str]:
    if optimizer not in LUDWIG_OPTIMIZER_TYPES:
        raise ConfigError(
            f'Task "{task_id}": unknown optimizer "{optimizer}" (available: {", ".join(LUDWIG_OPTIMIZER_TYPES)})'
        )
    return {"type": optimizer}


def _apply_class_weights(
    output_features: list[dict[str, Any]], class_counts: dict[str, int] | None, task_id: str
) -> list[dict[str, Any]]:
    """Balanced inverse-frequency weights for a category output loss, keyed by class NAME.

    Ludwig resolves a name-keyed class_weights dict against its own vocabulary at preprocessing
    time, so this never has to predict the index Ludwig assigns each class (Ludwig orders it by
    training-data frequency, which is not knowable before training).

    weight = total / (num_classes * class_count), the standard "balanced" formula (same as
    scikit-learn class_weight="balanced"): rare classes get a weight above 1.
    """
    if not class_counts:
        raise ConfigError(f'Task "{task_id}": class weighting requires a snapshot with a recorded class distribution')
    total = sum(class_counts.values())
    n = len(class_counts)
    weights = {name: total / (n * count) for name, count in class_counts.items()}
    return [
        {**f, "loss": {"type": "softmax_cross_entropy", "class_weights": weights}} if f["type"] == "category" else f
        for f in output_features
    ]


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    """The create-run/create-sweep form's knobs for this task: the shared trainer knobs (with this
    task's own defaults and bounds — an `ecd` task and an `llm` task have very different epoch and
    batch-size defaults) plus optimizer/validation-metric/image-size/class-weights, each offered
    only where it makes sense for this task. `LudwigHyperparameters`'s generic field introspection
    can't produce this: knob bounds are per-task, `batchSize` mixes numbers with `"auto"`, and
    `optimizer`/`validationMetric` are freeform strings with a curated menu rather than a fixed
    pydantic Literal.
    """
    spec = LUDWIG_TASKS.get(task.id)
    knobs = spec.trainer_knobs if spec is not None else None
    if knobs is None:
        return []

    learning = [
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=knobs.learning_rate.default, min=knobs.learning_rate.min, max=knobs.learning_rate.max,
            step=_learning_rate_step(knobs.learning_rate.min, knobs.learning_rate.max),
            description="How big a step the model takes each time it learns from some examples. Too high and it "
            "overshoots and never settles; too low and training is very slow. The default suits most cases.",
        ),
        ParamSpec(
            name="optimizer", label="Optimizer", type="choice", group=_TRAINING, default=None,
            choices=list(LUDWIG_OPTIMIZER_TYPES),
            description="The method the model uses to improve itself after each step. Leave it on Default unless "
            "you know you want a specific one; the default (Adam) is a safe general choice.",
        ),
    ]  # fmt: skip

    training = [
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING,
            default=knobs.epochs.default, min=knobs.epochs.min, max=knobs.epochs.max, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch). "
            "More epochs let it learn more, but too many can make it memorise the examples instead of learning "
            "the pattern.",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(knobs.batch_size.default), choices=[str(o) for o in knobs.batch_size.options],
            description="How many examples the model looks at before it updates itself. Bigger batches train faster "
            "but need more memory. \"auto\" lets the app pick the biggest size that fits.",
        ),
    ]  # fmt: skip

    stopping = [
        ParamSpec(
            name="earlyStopPatience", label="Early Stop Patience",
            description="Stops training automatically when results have not improved for this many epochs in a "
            "row, so time isn't wasted and the model doesn't over-learn your examples. Switch off to always "
            "train every epoch.",
            type="int", group=_STOPPING, default=knobs.early_stop_patience.default,
            min=knobs.early_stop_patience.min, step=1, disabled_value=-1,
        ),
    ]  # fmt: skip

    # Only non-LLM tasks (category or number output) have a comparable metric menu; experimental
    # tasks' text/sequence outputs don't.
    if task.status == "stable":
        metrics = _CLASSIFICATION_METRICS if task.annotation.requires_label_classes else _REGRESSION_METRICS
        stopping.append(
            ParamSpec(
                name="validationMetric", label="Early Stop / Best-Epoch Metric", type="choice", group=_STOPPING,
                default=None, choices=metrics,
                description="The score used to judge whether results are still improving and which epoch was the "
                "best one. Leave it on Default and the app picks a sensible score for your task.",
            )
        )  # fmt: skip

    # The head and the freeze switch only exist for ECD tasks (an LLM is fine-tuned as a whole).
    head: list[ParamSpec] = []
    if spec.model_type == "ecd":
        if any(e.pretrained for e in spec.encoders):
            head.append(
                ParamSpec(
                    name="freezeBackbone", label="Freeze backbone", type="bool", group=_MODEL, default=False,
                    description="Keeps the pretrained model exactly as it is and trains only the small part on top "
                    "of it. Faster and needs fewer examples, but it adapts less closely to your data.",
                )
            )  # fmt: skip
        head += [
            ParamSpec(
                name="headLayers", label="Head layers", type="int", group=_MODEL, default=0, min=0, max=4, step=1,
                description="Extra layers between the model and its final answer. 0 is the simplest option and is "
                "usually enough; try adding layers if results are poor and you have plenty of examples.",
            ),
            ParamSpec(
                name="headWidth", label="Head width", type="choice", group=_MODEL, default="256",
                choices=["64", "128", "256", "512"],
                description="How many units each extra layer has. Wider layers can pick up more complex patterns "
                "but are slower and more likely to memorise.",
            ),
            ParamSpec(
                name="headDropout", label="Head dropout", type="float", group=_MODEL, default=0.0, min=0.0, max=0.9,
                step=0.05,
                description="Randomly switches off this fraction of units while training, which stops the model "
                "from just memorising the examples. 0 means off.",
            ),
        ]  # fmt: skip

    data: list[ParamSpec] = []
    if task.modality == "text" and spec.model_type == "ecd":
        data.append(
            ParamSpec(
                name="maxSequenceLength", label="Max sequence length", type="choice", group=_DATA, default=None,
                choices=["64", "128", "256", "512"],
                description="Longer texts are cut off after this many tokens (roughly, pieces of words). A higher "
                "number keeps more of each text but is slower.",
            )
        )  # fmt: skip
    if task.modality == "vision":
        data.append(
            ParamSpec(
                name="imageSize", label="Image Size", type="choice", group=_DATA, default=None,
                choices=_IMAGE_SIZES,
                description="Every image is resized to a square of this many pixels before training. Larger keeps "
                "more detail but is slower and needs more memory.",
            )
        )  # fmt: skip

    if task.annotation.requires_label_classes:
        data.append(
            ParamSpec(
                name="useClassWeights",
                label="Weight classes by inverse frequency",
                type="bool",
                group=_DATA,
                default=False,
                description="Makes mistakes on rarer classes count for more, so the model doesn't simply favour the "
                "most common one. Recommended when some classes have far fewer examples than others.",
            )
        )

    # Section order of the form: the model's details, training, stopping, data.
    return [*head, *training, *learning, *stopping, *data]


def _learning_rate_step(lo: float | None, hi: float | None) -> float:
    span = (hi - lo) if lo is not None and hi is not None else 1.0
    return 0.0001 if span <= 0.01 else 0.001 if span <= 1 else 0.01
