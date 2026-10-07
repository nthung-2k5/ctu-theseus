"""Compile a task plus snapshot context plus hyperparameters into a YOLO training config.

The config is a plain dict stored verbatim as `training_runs.config`. `train.py` reads it back and
turns `train` into `YOLO.train()` keyword arguments, so everything that can be wrong with a request
(an unknown model, a bad size) surfaces here as a `ConfigError`, before a run row is committed.
"""

from typing import Any, Literal

from pydantic import Field

from theseus.backends.base import ConfigError, CustomModelRef, HyperparamsBase
from theseus.backends.yolo.tasks import IMAGE_SIZES, OPTIMIZERS, YOLO_TASKS, YoloTaskSpec
from theseus.schemas.common import ParamSpec
from theseus.services.task_registry import SnapshotContext, TaskDescriptor

# Section headings of the hyperparameter form, in the order it shows them.
_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_DATA = "Data"

# Ultralytics' `optimizer="auto"` ignores `lr0` and picks its own, so the learning rate would silently do
# nothing (and a sweep over it would be pointless). Always pass an explicit optimizer with an explicit rate.
DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EARLY_STOP_PATIENCE = 10


class YoloHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `model_id` (from HyperparamsBase, wire name `modelId`) is the id from the task's model catalog and
    defaults to the first.
    """

    # Square size every image is resized to. A multiple of 32.
    image_size: int | None = Field(default=None, ge=32, le=1280)
    optimizer: Literal["AdamW", "Adam", "SGD"] | None = None
    # False trains from the architecture alone, with randomly initialised weights.
    pretrained: bool | None = None
    # Number of leading layers kept frozen, which makes fine-tuning faster and needs fewer examples.
    freeze_layers: int | None = Field(default=None, ge=0, le=100)
    seed: int | None = Field(default=None, ge=0)


def _spec_for(task: TaskDescriptor) -> YoloTaskSpec:
    spec = YOLO_TASKS.get(task.id)
    if spec is None:
        raise ConfigError(f'The YOLO backend does not support the task "{task.id}"')
    return spec


def _column(ctx: SnapshotContext, kind: str) -> str:
    name = next((c.name for c in ctx.columns if c.kind == kind), None)
    if name is None:
        raise ConfigError(f'The snapshot has no "{kind}" column, which the YOLO backend needs')
    return name


def compile_yolo_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: YoloHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    if custom is not None:
        raise ConfigError("The YOLO backend does not support bring-your-own models")
    spec = _spec_for(task)
    if len(ctx.label_class_names) < 2:
        raise ConfigError("YOLO classification needs a snapshot with at least two classes")

    model_id = hp.model_id or spec.models[0].id
    if model_id not in {m.id for m in spec.models}:
        raise ConfigError(f'"{model_id}" is not a YOLO model for the task "{task.id}"')

    image_size = hp.image_size or spec.default_image_size
    if image_size % 32:
        raise ConfigError(f"Image size must be a multiple of 32, got {image_size}")

    pretrained = True if hp.pretrained is None else hp.pretrained
    batch = hp.batch_size if hp.batch_size is not None else spec.default_batch_size
    epochs = hp.epochs or spec.default_epochs
    patience = hp.early_stop_patience if hp.early_stop_patience is not None else DEFAULT_EARLY_STOP_PATIENCE

    train: dict[str, Any] = {
        "epochs": epochs,
        # -1 is Ultralytics' AutoBatch: the largest batch that fits on a GPU, the default size on a CPU.
        "batch": -1 if batch == "auto" else batch,
        "imgsz": image_size,
        "optimizer": hp.optimizer or DEFAULT_OPTIMIZER,
        "lr0": hp.learning_rate or DEFAULT_LEARNING_RATE,
        # -1 switches early stopping off: a patience of `epochs` can never be reached.
        "patience": epochs if patience < 0 else patience,
        "pretrained": pretrained,
    }
    if hp.freeze_layers:
        train["freeze"] = hp.freeze_layers
    if hp.seed is not None:
        train["seed"] = hp.seed

    return {
        "backend": "yolo",
        "task": spec.kind,
        "model": model_id,
        # A `.pt` is the pretrained checkpoint (downloaded on first use); a `.yaml` is the bare architecture.
        "weights": f"{model_id}.pt" if pretrained else f"{model_id}.yaml",
        # Snapshot parquet columns `data.py` reads, and the name the prediction frame's columns are keyed by.
        "imageColumn": _column(ctx, "storage_uri"),
        "labelColumn": _column(ctx, "label"),
        # Informational: the class order a trained model actually uses is its own (`model.names`), never this.
        "classNames": list(ctx.label_class_names),
        "train": train,
    }


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    """The create-run/create-sweep form's knobs. Hand-written because `batchSize` mixes numbers with
    `"auto"` and `imageSize`/`optimizer` are menus, none of which generic field introspection produces."""
    spec = YOLO_TASKS.get(task.id)
    if spec is None:
        return []

    return [
        ParamSpec(
            name="pretrained", label="Start from pretrained weights", type="bool", group=_MODEL, default=True,
            description="Starts from a model that has already learned general features from millions of images, "
            "which needs far fewer examples and trains faster. Turn off to train from scratch.",
        ),
        ParamSpec(
            name="freezeLayers", label="Freeze layers", type="int", group=_MODEL, default=0, min=0, max=100, step=1,
            description="Keeps this many of the first layers exactly as they are and trains only the rest. Faster "
            "and needs fewer examples, but it adapts less closely to your data. 0 trains everything.",
        ),
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING,
            default=spec.default_epochs, min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch). "
            "More epochs let it learn more, but too many can make it memorise the examples.",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(spec.default_batch_size), choices=["auto", "8", "16", "32", "64", "128"],
            description="How many examples the model looks at before it updates itself. Bigger batches train faster "
            "but need more memory. \"auto\" picks the biggest size that fits on the GPU.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=DEFAULT_LEARNING_RATE, min=0.00001, max=0.1, step=0.0001,
            description="How big a step the model takes each time it learns from some examples. Too high and it "
            "overshoots; too low and training is slow. SGD usually wants a larger value (around 0.01) than AdamW.",
        ),
        ParamSpec(
            name="optimizer", label="Optimizer", type="choice", group=_TRAINING,
            default=DEFAULT_OPTIMIZER, choices=list(OPTIMIZERS),
            description="The method the model uses to improve itself after each step. AdamW is a safe general choice.",
        ),
        ParamSpec(
            name="earlyStopPatience", label="Early Stop Patience", type="int", group=_STOPPING,
            default=DEFAULT_EARLY_STOP_PATIENCE, min=1, step=1, disabled_value=-1,
            description="Stops training automatically when results have not improved for this many epochs in a "
            "row. Switch off to always train every epoch.",
        ),
        ParamSpec(
            name="imageSize", label="Image Size", type="choice", group=_DATA,
            default=str(spec.default_image_size), choices=[str(s) for s in IMAGE_SIZES],
            description="Every image is resized to a square of this many pixels before training. Larger keeps "
            "more detail but is slower and needs more memory.",
        ),
    ]  # fmt: skip
