"""Compile a task plus snapshot context plus hyperparameters into a video training config.

The config is a plain dict stored verbatim as `training_runs.config`; `train.py` reads it back. Everything that can
be wrong with a request (an unknown model, a snapshot with one class) surfaces here as a `ConfigError`, before a
run row is committed. No torch import: this module is imported when the backend is merely listed.
"""

from typing import Any, Literal

from pydantic import Field

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, HyperparamsBase, seed_spec
from omnitrain.plugins.backends.video import tasks
from omnitrain.plugins.backends.video.custom import HF_VIDEO
from omnitrain.schemas.common import ParamSpec

_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_DATA = "Clips"


class VideoHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `model_id` (from HyperparamsBase, wire name `modelId`) is the id from `tasks.VIDEO_MODELS`, default the first.
    """

    # Frames per clip, and how many source frames apart they are taken. Together they set how much time one clip covers.
    num_frames: int | None = Field(default=None, ge=2, le=64)
    frame_stride: int | None = Field(default=None, ge=1, le=30)
    # Side of the square the network sees.
    frame_size: int | None = Field(default=None, ge=32, le=512)
    # Clips averaged per video when evaluating and serving; training sees one random clip per video per epoch.
    clips_per_video: int | None = Field(default=None, ge=1, le=10)
    optimizer: Literal["AdamW", "SGD"] | None = None
    pretrained: bool | None = None
    # Train only the new classification head, keeping the pretrained backbone fixed.
    freeze_backbone: bool | None = None
    horizontal_flip: bool | None = None


def compile_video_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: VideoHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    if task.id not in tasks.VIDEO_TASKS:
        raise ConfigError(f'The video backend does not support the task "{task.id}"')
    if len(ctx.label_class_names) < 2:
        raise ConfigError("Video classification needs a snapshot with at least two classes")

    if custom is None and hp.model_id and hp.model_id.startswith("custom:"):
        raise ConfigError(f'Custom model "{hp.model_id}" was not resolved (it may have been removed or disabled)')
    if custom is not None:
        if custom.kind != HF_VIDEO:
            raise ConfigError(f'A "{custom.kind}" custom model cannot be used for task "{task.id}"')
        # The files are not necessarily on disk yet, so the model's own clip (frames, size, normalisation) is read
        # from its config when training starts (train.py) and written into names.json; what is here is a stand-in.
        # Frames and size are the model's to decide: its position embeddings are tied to them.
        mean, std = tasks.VIDEO_MODELS[0].mean, tasks.VIDEO_MODELS[0].std
        model_id, num_frames, frame_size = hp.model_id or custom.id, 16, 224
    else:
        model_id = hp.model_id or tasks.VIDEO_MODELS[0].id
        model = tasks.MODELS_BY_ID.get(model_id)
        if model is None:
            raise ConfigError(f'"{model_id}" is not a video model for the task "{task.id}"')
        mean, std = model.mean, model.std
        num_frames, frame_size = hp.num_frames or model.frames, hp.frame_size or model.crop
    frame_stride = hp.frame_stride or tasks.DEFAULT_FRAME_STRIDE
    patience = hp.early_stop_patience if hp.early_stop_patience is not None else tasks.DEFAULT_EARLY_STOP_PATIENCE
    epochs = hp.epochs or tasks.DEFAULT_EPOCHS
    batch = hp.batch_size if isinstance(hp.batch_size, int) else tasks.DEFAULT_BATCH_SIZE

    return {
        "backend": "video",
        "task": "classify",
        "model": model_id,
        "pretrained": True if custom is not None or hp.pretrained is None else hp.pretrained,
        # Where the bring-your-own model's files are, for train.py. Absent for a built-in model.
        **({"custom": {"kind": custom.kind, "path": custom.local_path}} if custom is not None else {}),
        "clipColumn": _column(ctx, "storage_uri"),
        "labelColumn": _column(ctx, "label"),
        # The index a class gets is its position here. `train.py` writes this order into names.json, and a loaded
        # model reads it back from there: serving and export never consult this config or the label_classes table.
        "classNames": list(ctx.label_class_names),
        "clip": {
            "numFrames": num_frames,
            "frameStride": frame_stride,
            "frameSize": frame_size,
            "clipsPerVideo": hp.clips_per_video or tasks.DEFAULT_CLIPS_PER_VIDEO,
            "mean": list(mean),
            "std": list(std),
        },
        "train": {
            "epochs": epochs,
            "batchSize": batch,
            "learningRate": hp.learning_rate or tasks.DEFAULT_LEARNING_RATE,
            "optimizer": hp.optimizer or "AdamW",
            # -1 switches early stopping off: a patience of `epochs` can never be reached.
            "patience": epochs if patience < 0 else patience,
            "freezeBackbone": bool(hp.freeze_backbone),
            "horizontalFlip": True if hp.horizontal_flip is None else hp.horizontal_flip,
            "seed": hp.seed,
        },
    }


def _column(ctx: SnapshotContext, kind: str) -> str:
    name = next((c.name for c in ctx.columns if c.kind == kind), None)
    if name is None:
        raise ConfigError(f'The snapshot has no "{kind}" column, which the video backend needs')
    return name


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    """The create-run/create-sweep form's knobs. Hand-written because `batchSize` and the clip settings are menus."""
    if task.id not in tasks.VIDEO_TASKS:
        return []
    return [
        ParamSpec(
            name="pretrained", label="Start from pretrained weights", type="bool", group=_MODEL, default=True,
            description="Starts from a model that has already learned to recognise actions in 400 kinds of video "
            "(Kinetics-400), which needs far fewer clips and trains much faster. Turn off to train from scratch.",
        ),
        ParamSpec(
            name="freezeBackbone", label="Freeze backbone", type="bool", group=_MODEL, default=False,
            description="Trains only the final classification layer and leaves the rest exactly as pretrained. "
            "Much faster and needs very few clips, but it adapts less closely to your data.",
        ),
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING,
            default=tasks.DEFAULT_EPOCHS, min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch).",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(tasks.DEFAULT_BATCH_SIZE), choices=["1", "2", "4", "8", "16"],
            description="How many clips the model looks at before it updates itself. Video is memory-hungry, so "
            "small batches are normal; lower this first if training runs out of GPU memory.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=tasks.DEFAULT_LEARNING_RATE, min=0.00001, max=0.1, step=0.0001,
            description="How big a step the model takes each time it learns from some clips. Too high and it "
            "overshoots; too low and training is slow.",
        ),
        seed_spec(_TRAINING),
        ParamSpec(
            name="optimizer", label="Optimizer", type="choice", group=_TRAINING,
            default="AdamW", choices=list(tasks.OPTIMIZERS),
            description="The method the model uses to improve itself after each step. AdamW is a safe choice.",
        ),
        ParamSpec(
            name="horizontalFlip", label="Random horizontal flip", type="bool", group=_TRAINING, default=True,
            description="Randomly mirrors training clips left to right so the model does not depend on which side "
            "something happens. Turn off when direction matters (for example, turning left versus right).",
        ),
        ParamSpec(
            name="earlyStopPatience", label="Early Stop Patience", type="int", group=_STOPPING,
            default=tasks.DEFAULT_EARLY_STOP_PATIENCE, min=1, step=1, disabled_value=-1,
            description="Stops training when validation results have not improved for this many epochs in a row.",
        ),
        ParamSpec(
            name="numFrames", label="Frames per clip", type="choice", group=_DATA,
            default="16", choices=[str(n) for n in tasks.FRAME_COUNTS],
            description="How many frames the model sees at once. More frames capture slower actions but cost more "
            "memory and time. A Hugging Face model uses its own number and ignores this.",
        ),
        ParamSpec(
            name="frameStride", label="Frame stride", type="int", group=_DATA,
            default=tasks.DEFAULT_FRAME_STRIDE, min=1, max=30, step=1,
            description="Takes every Nth frame of the video. A clip covers frames-per-clip x stride frames of the "
            "video, so raise this for slow actions or high frame rates.",
        ),
        ParamSpec(
            name="frameSize", label="Frame size", type="choice", group=_DATA,
            default="112", choices=[str(n) for n in tasks.FRAME_SIZES],
            description="Every frame is resized so its shorter side is this many pixels, then cropped to a square. "
            "Larger keeps more detail but is much slower. Each model was pretrained at 112 or 224. A Hugging Face "
            "model uses its own size and ignores this.",
        ),
        ParamSpec(
            name="clipsPerVideo", label="Clips per video when scoring", type="int", group=_DATA,
            default=tasks.DEFAULT_CLIPS_PER_VIDEO, min=1, max=10, step=1,
            description="When evaluating or predicting, this many clips spread across the video are scored and "
            "their predictions averaged. Training always sees one random clip per video per epoch.",
        ),
    ]  # fmt: skip
