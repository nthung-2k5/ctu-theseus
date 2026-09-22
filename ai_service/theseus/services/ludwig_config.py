"""Compile a task descriptor plus snapshot context plus user selections into a Ludwig config.

Ported from server/lib/ludwig/{compile,schema,serialize}.ts. The compiled config is validated
here, before a run row is ever committed, so an invalid configuration surfaces as a 400 rather
than as a worker crash discovered later.
"""

import copy
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from theseus import constants as C
from theseus.schemas.common import ApiModel
from theseus.services.task_registry import (
    LUDWIG_OPTIMIZER_TYPES,
    EncoderChoice,
    SnapshotContext,
    TaskDescriptor,
)

LUDWIG_VERSION = "0.17.5"


class ConfigError(ValueError):
    """A user-correctable problem with the requested training configuration."""


class TrainerSelections(ApiModel):
    """User-facing hyperparameter choices (camelCase on the wire, and in runs.hyperparameters)."""

    epochs: int | None = None
    batch_size: int | Literal["auto"] | None = None
    learning_rate: float | None = None
    early_stop_patience: int | None = None
    # Encoder id from the task descriptor encoders list. Defaults to the first.
    encoder_id: str | None = None
    # Weight each class loss inversely to its snapshot frequency (category outputs only).
    use_class_weights: bool | None = None
    # There is deliberately no augmentation here: it moved to snapshot creation, where the augmented
    # copies are real, browsable train-split items (see services/augmentation.py).
    # Square-resize every image input feature to this many pixels per side. Ignored for non-vision tasks.
    image_size: int | None = None
    # Metric tracked for early stopping / best epoch instead of Ludwig per-output-type default.
    # Passed straight through: Ludwig itself rejects a name the output feature type does not support.
    validation_metric: str | None = None
    # Defaults to Ludwig per-model-type default (Adam for ECD) when unset.
    optimizer: str | None = None


# -- Validation schema (replaces the zod schema) ---------------------------------------------


class _Feature(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: Literal["image", "text", "audio", "number", "category", "binary", "sequence", "vector"]
    column: str


class _Typed(BaseModel):
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
    preprocessing: _Preprocessing
    trainer: _Trainer
    ludwig_version: str


# -- Compiler --------------------------------------------------------------------------------


def compile_ludwig_config(
    task: TaskDescriptor, ctx: SnapshotContext, selections: TrainerSelections | None = None
) -> dict[str, Any]:
    sel = selections or TrainerSelections()
    if task.ludwig is None:
        raise ConfigError(f'Task "{task.id}" has no Ludwig backend (status: {task.status})')
    ludwig = task.ludwig
    knobs = ludwig.trainer_knobs

    declared = copy.deepcopy(ludwig.input_features)
    if declared:
        input_features = [
            _with_image_resize(_with_encoder(f, ludwig.encoders, sel.encoder_id), sel.image_size) for f in declared
        ]
    else:
        # Tabular tasks do not know their column names statically: derive one number feature
        # per scalar column the snapshot actually contains.
        input_features = [
            {"name": c.name, "type": "number", "column": c.name} for c in ctx.columns if c.kind == "scalar"
        ]

    if not input_features:
        raise ConfigError(f'Task "{task.id}": no input features could be derived from the snapshot columns')

    output_features = copy.deepcopy(ludwig.output_features)
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
        "model_type": ludwig.model_type,
        "input_features": input_features,
        "output_features": output_features,
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


def _with_encoder(feature: dict[str, Any], encoders: list[EncoderChoice], encoder_id: str | None) -> dict[str, Any]:
    if feature.get("encoder") or not encoders:
        return feature
    if encoder_id:
        encoder = next((e for e in encoders if e.id == encoder_id), None)
    else:
        encoder = encoders[0]
    if encoder is None:
        available = ", ".join(e.id for e in encoders)
        raise ConfigError(f'Unknown encoder "{encoder_id}" (available: {available})')
    return {
        **feature,
        "encoder": {"type": encoder.encoder_type, "use_pretrained": encoder.pretrained, **(encoder.params or {})},
    }


def _with_image_resize(feature: dict[str, Any], size: int | None) -> dict[str, Any]:
    """Square-resize an image input feature. A no-op for every non-image feature or unset size."""
    if not size or feature["type"] != "image":
        return feature
    return {**feature, "preprocessing": {**feature.get("preprocessing", {}), "height": size, "width": size}}


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
