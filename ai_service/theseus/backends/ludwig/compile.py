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
from theseus.backends.base import ConfigError, HyperparamsBase
from theseus.backends.ludwig.tasks import LUDWIG_OPTIMIZER_TYPES, LUDWIG_TASKS, EncoderChoice
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
_OPTIMISATION = "Optimisation"
_STOPPING = "Batching & stopping"
_HEAD = "Backbone & head"
_DATA = "Data & loss"


class LudwigHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `model_id` (from HyperparamsBase) is the encoder id from the task's encoder catalog; defaults
    to the first. Kept as `encoderId` on the wire (the field predates the generic `model_id` name
    and every existing client already sends it) rather than picking up HyperparamsBase's generated
    `modelId` alias. There is deliberately no augmentation here: it moved to snapshot creation,
    where the augmented copies are real, browsable train-split items (see services/augmentation.py).
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
    input_features: list[_InputFeature] = Field(min_length=1)
    output_features: list[_OutputFeature] = Field(min_length=1)
    combiner: _Combiner | None = None
    preprocessing: _Preprocessing
    trainer: _Trainer
    ludwig_version: str


# -- Compiler --------------------------------------------------------------------------------


def compile_ludwig_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: LudwigHyperparameters | None = None
) -> dict[str, Any]:
    sel = hp or LudwigHyperparameters()
    spec = LUDWIG_TASKS.get(task.id)
    if spec is None:
        raise ConfigError(f'Task "{task.id}" has no Ludwig backend (status: {task.status})')
    knobs = spec.trainer_knobs

    if sel.freeze_backbone and not any(e.pretrained for e in spec.encoders):
        raise ConfigError(f'Task "{task.id}" has no pretrained backbone to freeze')
    combiner = _head_combiner(sel, task.id, spec.model_type)

    declared = copy.deepcopy(spec.input_features)
    if declared:
        input_features = [
            _with_sequence_length(
                _with_image_resize(_with_encoder(f, spec.encoders, sel.model_id, sel.freeze_backbone), sel.image_size),
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


def _with_encoder(
    feature: dict[str, Any], encoders: list[EncoderChoice], model_id: str | None, freeze: bool | None = None
) -> dict[str, Any]:
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

    optimisation = [
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_OPTIMISATION,
            default=knobs.learning_rate.default, min=knobs.learning_rate.min, max=knobs.learning_rate.max,
            step=_learning_rate_step(knobs.learning_rate.min, knobs.learning_rate.max),
        ),
        ParamSpec(
            name="optimizer", label="Optimizer", type="choice", group=_OPTIMISATION, default=None,
            choices=list(LUDWIG_OPTIMIZER_TYPES),
            description="Defaults to Ludwig's per-model-type default (Adam for ECD)",
        ),
    ]  # fmt: skip

    stopping = [
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_STOPPING,
            default=knobs.epochs.default, min=knobs.epochs.min, max=knobs.epochs.max, step=1,
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_STOPPING,
            default=str(knobs.batch_size.default), choices=[str(o) for o in knobs.batch_size.options],
        ),
        ParamSpec(
            name="earlyStopPatience", label="Early Stop Patience", description="-1 disables early stopping",
            type="int", group=_STOPPING, default=knobs.early_stop_patience.default,
            min=knobs.early_stop_patience.min, step=1,
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
            )
        )  # fmt: skip

    # The head and the freeze switch only exist for ECD tasks (an LLM is fine-tuned as a whole).
    head: list[ParamSpec] = []
    if spec.model_type == "ecd":
        if any(e.pretrained for e in spec.encoders):
            head.append(
                ParamSpec(
                    name="freezeBackbone", label="Freeze backbone", type="bool", group=_HEAD, default=False,
                    description="Trains only the head and keeps the pretrained weights fixed",
                )
            )  # fmt: skip
        head += [
            ParamSpec(
                name="headLayers", label="Head layers", type="int", group=_HEAD, default=0, min=0, max=4, step=1,
                description="Hidden layers between the backbone and the output. 0 is a plain linear head.",
            ),
            ParamSpec(
                name="headWidth", label="Head width", type="choice", group=_HEAD, default="256",
                choices=["64", "128", "256", "512"], description="Units per hidden layer of the head.",
            ),
            ParamSpec(
                name="headDropout", label="Head dropout", type="float", group=_HEAD, default=0.0, min=0.0, max=0.9,
                step=0.05, description="Dropout applied inside the head's hidden layers.",
            ),
        ]  # fmt: skip

    data: list[ParamSpec] = []
    if task.modality == "text" and spec.model_type == "ecd":
        data.append(
            ParamSpec(
                name="maxSequenceLength", label="Max sequence length", type="choice", group=_DATA, default=None,
                choices=["64", "128", "256", "512"], description="Truncates each text to this many tokens",
            )
        )  # fmt: skip
    if task.modality == "vision":
        data.append(
            ParamSpec(
                name="imageSize", label="Image Size", type="choice", group=_DATA, default=None,
                choices=_IMAGE_SIZES, description="Resizes every training image to a square of this size",
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
                description="Balances the loss so a minority class isn't drowned out by a majority one — "
                "recommended for imbalanced datasets",
            )
        )

    # Section order of the form: optimisation, batching and stopping, backbone and head, data and loss.
    return [*optimisation, *stopping, *head, *data]


def _learning_rate_step(lo: float | None, hi: float | None) -> float:
    span = (hi - lo) if lo is not None and hi is not None else 1.0
    return 0.0001 if span <= 0.01 else 0.001 if span <= 1 else 0.01
