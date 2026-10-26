"""Compile a task plus snapshot context plus hyperparameters into a PyTorch training config.

The config is a plain dict stored verbatim as `training_runs.config`; `train.py` reads it back. Everything that can be
wrong with a request (an unknown model, a snapshot with one class) surfaces here as a `ConfigError`, before a run row is
committed. No torch import: this module is imported when the backend is merely listed.

`inputKind` in the config says what a row is (`image`, `tabular` or `audio`) and so which of `train.py`'s flows
runs. A knob that only makes sense for the other kind is refused rather than ignored, so a sweep over it measures
something.
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
    mixed_precision_spec,
    refuse_unsupported,
    seed_spec,
    warmup_spec,
    weight_decay_spec,
)
from omnitrain.plugins.backends.pytorch import tasks
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, ImageSpec
from omnitrain.schemas.common import ParamSpec

_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_DATA = "Images"
_AUDIO = "Audio"
# The narrowest spectrogram (in frames) the networks accept: a ResNet halves each side five times.
MIN_FRAMES = 32

# Knobs of some kinds of input, refused for the others. Starting from pretrained weights suits images and sounds.
_IMAGE_ONLY = ("image_size", "horizontal_flip")
_PRETRAINED = ("pretrained", "freeze_backbone")
_TABULAR_ONLY = ("dropout", "use_class_weights")
_AUDIO_ONLY = ("sample_rate", "clip_seconds", "n_mels", "clips_per_audio", "spec_augment")


class PytorchHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `model_id` (from HyperparamsBase, wire name `modelId`) is an id from the task's catalog in `tasks`, default the
    first.
    """

    # Images: side of the square the network sees (a vision transformer only works at its own size).
    image_size: int | None = Field(default=None, ge=32, le=1024)
    pretrained: bool | None = None
    # Train only the new classification head, keeping the pretrained backbone fixed.
    freeze_backbone: bool | None = None
    horizontal_flip: bool | None = None
    optimizer: Literal["AdamW", "SGD"] | None = None
    # Tables: the dropout probability between hidden layers, and weighting each class inversely to its frequency.
    dropout: float | None = Field(default=None, ge=0, lt=1)
    use_class_weights: bool | None = None
    # Sounds: the rate every clip is resampled to, the length of one window, the mel bands of its spectrogram, how many
    # windows of a longer clip are scored (their probabilities averaged), and masking bands of a training spectrogram.
    sample_rate: int | None = Field(default=None, ge=4000, le=48000)
    clip_seconds: float | None = Field(default=None, ge=1, le=60)
    n_mels: int | None = Field(default=None, ge=16, le=256)
    clips_per_audio: int | None = Field(default=None, ge=1, le=10)
    spec_augment: bool | None = None


def _refuse(hp: PytorchHyperparameters, task: TaskDescriptor, *names: str) -> None:
    try:
        refuse_unsupported(hp, "PyTorch", *names)
    except ConfigError as e:
        raise ConfigError(f'{e} for the task "{task.id}"') from None


def compile_pytorch_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    if task.id not in tasks.IMAGE_TASKS | tasks.AUDIO_TASKS and task.id not in tasks.TABULAR_TASKS:
        raise ConfigError(f'The PyTorch backend does not support the task "{task.id}"')
    if custom is not None or (hp.model_id and hp.model_id.startswith("custom:")):
        raise ConfigError("The PyTorch backend does not take bring-your-own models yet")
    if task.id in tasks.IMAGE_TASKS:
        _refuse(hp, task, *_TABULAR_ONLY, *_AUDIO_ONLY)
        return _compile_image(task, ctx, hp)
    if task.id in tasks.AUDIO_TASKS:
        _refuse(hp, task, *_TABULAR_ONLY, *_IMAGE_ONLY)
        return _compile_audio(task, ctx, hp)
    _refuse(hp, task, *_IMAGE_ONLY, *_PRETRAINED, *_AUDIO_ONLY)
    return _compile_tabular(task, ctx, hp)


def _train_block(hp: PytorchHyperparameters, defaults: dict[str, Any], *, epochs: int) -> dict[str, Any]:
    """The `train` section both kinds share. Unset optimisation knobs keep what each optimizer already did (see
    loop.make_optimizer); the rest are off by default."""
    patience = hp.early_stop_patience if hp.early_stop_patience is not None else defaults["patience"]
    return {
        "epochs": epochs,
        "batchSize": hp.batch_size if isinstance(hp.batch_size, int) else defaults["batch"],
        "learningRate": hp.learning_rate or defaults["learning_rate"],
        "optimizer": hp.optimizer or "AdamW",
        # -1 switches early stopping off: a patience of `epochs` can never be reached.
        "patience": epochs if patience < 0 else patience,
        "seed": hp.seed,
        "weightDecay": hp.weight_decay,
        "lrSchedule": hp.lr_schedule or "constant",
        "warmupEpochs": hp.warmup_epochs or 0,
        "gradClipNorm": hp.grad_clip_norm,
        "mixedPrecision": bool(hp.mixed_precision),
    }


def _compile_image(task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters) -> dict[str, Any]:
    if len(ctx.label_class_names) < 2:
        raise ConfigError("Image classification needs a snapshot with at least two classes")
    model_id = hp.model_id or tasks.IMAGE_MODELS[0].id
    model = tasks.MODELS_BY_ID.get(model_id)
    if model is None:
        raise ConfigError(f'"{model_id}" is not an image model for the task "{task.id}"')
    size = hp.image_size or model.size
    if model.fixed_size and size != model.size:
        raise ConfigError(f"{model.label} only works at {model.size} pixels, not {size}")

    defaults = {
        "epochs": tasks.DEFAULT_EPOCHS,
        "batch": tasks.DEFAULT_BATCH_SIZE,
        "learning_rate": tasks.DEFAULT_LEARNING_RATE,
        "patience": tasks.DEFAULT_EARLY_STOP_PATIENCE,
    }
    spec = ImageSpec(size=size, mean=tasks.IMAGENET_MEAN, std=tasks.IMAGENET_STD)
    return {
        "backend": "pytorch",
        "inputKind": "image",
        "task": "classify",
        "model": model_id,
        "pretrained": True if hp.pretrained is None else hp.pretrained,
        "imageColumn": _column(ctx, "storage_uri"),
        "labelColumn": _column(ctx, "label"),
        # The index a class gets is its position here. `train.py` writes this order into names.json, and a loaded
        # model reads it back from there: serving and export never consult this config or the label_classes table.
        "classNames": list(ctx.label_class_names),
        "image": spec.to_meta(),
        "train": {
            **_train_block(hp, defaults, epochs=hp.epochs or defaults["epochs"]),
            "freezeBackbone": bool(hp.freeze_backbone),
            "horizontalFlip": True if hp.horizontal_flip is None else hp.horizontal_flip,
        },
    }


def _compile_audio(task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters) -> dict[str, Any]:
    if len(ctx.label_class_names) < 2:
        raise ConfigError("Audio classification needs a snapshot with at least two classes")
    model_id = hp.model_id or tasks.AUDIO_MODELS[0].id
    model = tasks.AUDIO_MODELS_BY_ID.get(model_id)
    if model is None:
        raise ConfigError(f'"{model_id}" is not an audio model for the task "{task.id}"')
    pretrained = model.pretrained if hp.pretrained is None else hp.pretrained
    if pretrained and not model.pretrained:
        raise ConfigError(f"{model.label} has no pretrained weights to start from")

    d = tasks.AUDIO_DEFAULTS
    # The mean and std are placeholders: `train.py` fits them on the train clips and writes the real ones to names.json.
    spec = AudioSpec(
        sample_rate=hp.sample_rate or d["sample_rate"],
        seconds=hp.clip_seconds or d["clip_seconds"],
        n_mels=hp.n_mels or d["n_mels"],
        clips=hp.clips_per_audio or d["clips_per_audio"],
    )
    if spec.frames < MIN_FRAMES:
        raise ConfigError(
            f"A window of {spec.seconds:g} s at {spec.sample_rate} Hz gives a spectrogram only {max(spec.frames, 0)} "
            f"frames wide; the network needs at least {MIN_FRAMES}. Use a longer clip or a higher sample rate"
        )
    return {
        "backend": "pytorch",
        "inputKind": "audio",
        "task": "classify",
        "model": model_id,
        "pretrained": pretrained,
        "audioColumn": _column(ctx, "storage_uri"),
        "labelColumn": _column(ctx, "label"),
        # The index a class gets is its position here. `train.py` writes this order into names.json, and a loaded
        # model reads it back from there: serving and export never consult this config or the label_classes table.
        "classNames": list(ctx.label_class_names),
        "audio": spec.to_meta(),
        "train": {
            **_train_block(hp, d, epochs=hp.epochs or d["epochs"]),
            "freezeBackbone": bool(hp.freeze_backbone),
            "specAugment": True if hp.spec_augment is None else hp.spec_augment,
        },
    }


def _compile_tabular(task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters) -> dict[str, Any]:
    kind = tasks.TABULAR_TASKS[task.id]
    if kind == "classify" and len(ctx.label_class_names) < 2:
        raise ConfigError("Classification needs a snapshot with at least two classes")
    if kind == "regress":
        _refuse(hp, task, "use_class_weights")
    model_id = hp.model_id or tasks.TABULAR_MODELS[0].id
    model = tasks.TABULAR_MODELS_BY_ID.get(model_id)
    if model is None:
        raise ConfigError(f'"{model_id}" is not a table model for the task "{task.id}"')
    # Tabular tasks do not know their column names statically: every scalar column of the snapshot is a feature.
    columns = [c.name for c in ctx.columns if c.kind == "scalar"]
    if not columns:
        raise ConfigError(f'Task "{task.id}": the snapshot has no numeric columns to learn from')

    d = tasks.TABULAR_DEFAULTS
    defaults = {k: d[k] for k in ("batch", "learning_rate", "patience")}
    return {
        "backend": "pytorch",
        "inputKind": "tabular",
        "task": kind,
        "model": model_id,
        "inputColumns": columns,
        "labelColumn": _column(ctx, "label"),
        # Informational: the class order a trained model uses is the one `train.py` writes into names.json.
        "classNames": list(ctx.label_class_names) if kind == "classify" else [],
        "network": {"hidden": list(model.hidden), "dropout": d["dropout"] if hp.dropout is None else hp.dropout},
        "train": {
            **_train_block(hp, defaults, epochs=hp.epochs or d["epochs"]),
            "useClassWeights": bool(hp.use_class_weights) and kind == "classify",
        },
    }


def _column(ctx: SnapshotContext, kind: str) -> str:
    name = next((c.name for c in ctx.columns if c.kind == kind), None)
    if name is None:
        raise ConfigError(f'The snapshot has no "{kind}" column, which the PyTorch backend needs')
    return name


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    """The create-run/create-sweep form's knobs. Hand-written because `batchSize` and `imageSize` are menus."""
    if task.id in tasks.IMAGE_TASKS:
        return _image_specs()
    if task.id in tasks.TABULAR_TASKS:
        return _tabular_specs(tasks.TABULAR_TASKS[task.id])
    if task.id in tasks.AUDIO_TASKS:
        return _audio_specs()
    return []


def _audio_specs() -> list[ParamSpec]:
    d = tasks.AUDIO_DEFAULTS
    return [
        ParamSpec(
            name="pretrained", label="Start from pretrained weights", type="bool", group=_MODEL, default=True,
            description="Starts from a network that has already learned to see patterns in a million photos "
            "(ImageNet). A spectrogram is an image of a sound, and those patterns carry over, so it needs far fewer "
            "recordings. Turn off to train from scratch; the small CNN always does.",
        ),
        ParamSpec(
            name="freezeBackbone", label="Freeze backbone", type="bool", group=_MODEL, default=False,
            description="Trains only the final classification layer and leaves the rest exactly as it started. Much "
            "faster and needs very few recordings, but it adapts less closely to your sounds.",
        ),
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING, default=d["epochs"], min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch).",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(d["batch"]), choices=[str(n) for n in tasks.AUDIO_BATCH_SIZES],
            description="How many recordings the model looks at before it updates itself. Bigger batches are "
            "faster but need more GPU memory; lower this first if training runs out of memory.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=d["learning_rate"], min=0.00001, max=0.1, step=0.0001,
            description="How big a step the model takes each time it learns from some recordings. Too high and it "
            "overshoots; too low and training is slow.",
        ),
        seed_spec(_TRAINING),
        weight_decay_spec(_TRAINING),
        lr_schedule_spec(_TRAINING),
        warmup_spec(_TRAINING),
        grad_clip_spec(_TRAINING),
        mixed_precision_spec(_TRAINING),
        _optimizer_spec(),
        ParamSpec(
            name="specAugment", label="Mask parts of the spectrogram", type="bool", group=_TRAINING, default=True,
            description="While training, blanks out a few random bands of pitch and stretches of time, so the model "
            "learns to recognise a sound from whatever part of it is audible. Helps most with few recordings.",
        ),
        _patience_spec(d["patience"]),
        ParamSpec(
            name="sampleRate", label="Sample rate (Hz)", type="choice", group=_AUDIO,
            default=str(d["sample_rate"]), choices=[str(n) for n in tasks.AUDIO_SAMPLE_RATES],
            description="Every recording is converted to this many samples per second. Higher keeps more of the high "
            "pitches but costs more; 16000 is enough for speech and most everyday sounds.",
        ),
        ParamSpec(
            name="clipSeconds", label="Clip length (seconds)", type="float", group=_AUDIO,
            default=d["clip_seconds"], min=1, max=60, step=0.5,
            description="The model listens to this many seconds at a time. A shorter recording is padded with "
            "silence; a longer one is cut (at a random place while training). Pick about the length of the sound "
            "you want to recognise.",
        ),
        ParamSpec(
            name="nMels", label="Mel bands", type="choice", group=_AUDIO,
            default=str(d["n_mels"]), choices=[str(n) for n in tasks.AUDIO_MEL_BANDS],
            description="How finely the pitch is divided in the spectrogram. More bands show more detail and cost "
            "more.",
        ),
        ParamSpec(
            name="clipsPerAudio", label="Windows per recording when scoring", type="int", group=_AUDIO,
            default=d["clips_per_audio"], min=1, max=10, step=1,
            description="When evaluating or predicting a recording longer than the clip length, this many windows "
            "spread across it are scored and their predictions averaged. Training sees one random window per "
            "recording per epoch.",
        ),
    ]  # fmt: skip


def _image_specs() -> list[ParamSpec]:
    return [
        ParamSpec(
            name="pretrained", label="Start from pretrained weights", type="bool", group=_MODEL, default=True,
            description="Starts from a model that has already learned to see on a million everyday photos "
            "(ImageNet), which needs far fewer of your images and trains much faster. Turn off to train from "
            "scratch.",
        ),
        ParamSpec(
            name="freezeBackbone", label="Freeze backbone", type="bool", group=_MODEL, default=False,
            description="Trains only the final classification layer and leaves the rest exactly as pretrained. "
            "Much faster and needs very few images, but it adapts less closely to your data.",
        ),
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING,
            default=tasks.DEFAULT_EPOCHS, min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch).",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(tasks.DEFAULT_BATCH_SIZE), choices=["4", "8", "16", "32", "64", "128"],
            description="How many images the model looks at before it updates itself. Bigger batches are faster "
            "but need more GPU memory; lower this first if training runs out of memory.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=tasks.DEFAULT_LEARNING_RATE, min=0.00001, max=0.1, step=0.0001,
            description="How big a step the model takes each time it learns from some images. Too high and it "
            "overshoots; too low and training is slow.",
        ),
        seed_spec(_TRAINING),
        weight_decay_spec(_TRAINING),
        lr_schedule_spec(_TRAINING),
        warmup_spec(_TRAINING),
        grad_clip_spec(_TRAINING),
        mixed_precision_spec(_TRAINING),
        _optimizer_spec(),
        ParamSpec(
            name="horizontalFlip", label="Random horizontal flip", type="bool", group=_TRAINING, default=True,
            description="Randomly mirrors training images left to right so the model does not depend on which side "
            "something is on. Turn off when direction matters (for example, text or left versus right hands).",
        ),
        _patience_spec(tasks.DEFAULT_EARLY_STOP_PATIENCE),
        ParamSpec(
            name="imageSize", label="Image size", type="choice", group=_DATA,
            default=str(tasks.IMAGE_SIZES[3]), choices=[str(n) for n in tasks.IMAGE_SIZES],
            description="Every image is resized so its shorter side is this many pixels, then cropped to a square. "
            "Larger keeps more detail but is much slower. Each model was pretrained at 224, and a vision "
            "transformer only works at 224.",
        ),
    ]  # fmt: skip


def _tabular_specs(kind: str) -> list[ParamSpec]:
    d = tasks.TABULAR_DEFAULTS
    specs = [
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING, default=d["epochs"], min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch).",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(d["batch"]), choices=[str(n) for n in tasks.TABULAR_BATCH_SIZES],
            description="How many rows the model looks at before it updates itself. Smaller batches update more "
            "often and add noise that can help on a small table; larger ones are faster.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=d["learning_rate"], min=0.00001, max=0.1, step=0.0001,
            description="How big a step the model takes each time it learns from some rows. Too high and it "
            "overshoots; too low and training is slow.",
        ),
        seed_spec(_TRAINING),
        weight_decay_spec(_TRAINING),
        lr_schedule_spec(_TRAINING),
        warmup_spec(_TRAINING),
        grad_clip_spec(_TRAINING),
        mixed_precision_spec(_TRAINING),
        _optimizer_spec(),
        ParamSpec(
            name="dropout", label="Dropout", type="float", group=_MODEL, default=d["dropout"], min=0, max=0.9,
            step=0.05,
            description="Randomly switches off this share of the hidden units while training, so the model cannot "
            "lean on any one of them. Raise it if the model memorises the training rows (validation results get "
            "worse while training results keep improving).",
        ),
    ]  # fmt: skip
    if kind == "classify":
        specs.append(
            ParamSpec(
                name="useClassWeights",
                label="Balance classes",
                type="bool",
                group=_TRAINING,
                default=False,
                description="Weights each class inversely to how often it appears, so a rare class counts as much as "
                "a common one. Helps when one class is far more common than the others, at the cost of some "
                "accuracy on the common one.",
            )
        )
    specs.append(_patience_spec(d["patience"]))
    return specs


def _optimizer_spec() -> ParamSpec:
    return ParamSpec(
        name="optimizer", label="Optimizer", type="choice", group=_TRAINING,
        default="AdamW", choices=list(tasks.OPTIMIZERS),
        description="The method the model uses to improve itself after each step. AdamW is a safe choice.",
    )  # fmt: skip


def _patience_spec(default: int) -> ParamSpec:
    return ParamSpec(
        name="earlyStopPatience", label="Early Stop Patience", type="int", group=_STOPPING,
        default=default, min=1, step=1, disabled_value=-1,
        description="Stops training when validation results have not improved for this many epochs in a row.",
    )  # fmt: skip
