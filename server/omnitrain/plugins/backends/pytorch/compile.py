"""Compile a task plus snapshot context plus hyperparameters into a PyTorch training config.

The config is a plain dict stored verbatim as `training_runs.config`; `train.py` reads it back. Everything that can be
wrong with a request (an unknown model, a snapshot with one class) surfaces here as a `ConfigError`, before a run row is
committed. No torch import: this module is imported when the backend is merely listed.

`inputKind` in the config says what a row is (`image`, `tabular`, `audio` or `text`) and so which of `train.py`'s flows
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
from omnitrain.plugins.backends.pytorch.custom import HF_IMAGE, HF_TEXT
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, ImageSpec, TextSpec
from omnitrain.schemas.common import ParamSpec

_MODEL = "Model details"
_TRAINING = "Training"
_STOPPING = "Stopping"
_DATA = "Images"
_AUDIO = "Audio"
_TEXT = "Text"
# The narrowest spectrogram (in frames) the networks accept: a ResNet halves each side five times.
MIN_FRAMES = 32

# Knobs of some kinds of input, refused for the others. Starting from pretrained weights suits images and sounds.
_IMAGE_ONLY = ("image_size", "horizontal_flip")
_PRETRAINED = ("pretrained", "freeze_backbone")
_TABULAR_ONLY = ("dropout", "use_class_weights")
_AUDIO_ONLY = ("sample_rate", "clip_seconds", "n_mels", "clips_per_audio", "spec_augment")
_TEXT_ONLY = ("max_tokens", "vocab_size", "min_word_count")


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
    # Text: the most words of a text the model reads (the rest is cut), the most distinct words it learns, and how
    # often a word must occur in the training texts to be learned (a rarer one is read as unknown).
    max_tokens: int | None = Field(default=None, ge=8, le=1024)
    vocab_size: int | None = Field(default=None, ge=100, le=200000)
    min_word_count: int | None = Field(default=None, ge=1, le=1000)


# The one bring-your-own kind each input kind takes.
_CUSTOM_KIND = {"image": HF_IMAGE, "text": HF_TEXT}


def _input_kind(task: TaskDescriptor) -> str:
    if task.id in tasks.IMAGE_TASKS:
        return "image"
    if task.id in tasks.TEXT_TASKS:
        return "text"
    return "other"


def _refuse(hp: PytorchHyperparameters, task: TaskDescriptor, *names: str) -> None:
    try:
        refuse_unsupported(hp, "PyTorch", *names)
    except ConfigError as e:
        raise ConfigError(f'{e} for the task "{task.id}"') from None


def compile_pytorch_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    supported = tasks.IMAGE_TASKS | tasks.AUDIO_TASKS | tasks.TEXT_TASKS | tasks.TOKEN_TASKS
    if task.id not in supported and task.id not in tasks.TABULAR_TASKS:
        raise ConfigError(f'The PyTorch backend does not support the task "{task.id}"')
    if custom is None and hp.model_id and hp.model_id.startswith("custom:"):
        raise ConfigError(f'Custom model "{hp.model_id}" was not resolved (it may have been removed or disabled)')
    if custom is not None and custom.kind != _CUSTOM_KIND.get(_input_kind(task)):
        raise ConfigError(f'A "{custom.kind}" custom model cannot be used for task "{task.id}"')
    if task.id in tasks.TOKEN_TASKS:
        _refuse(hp, task, *_IMAGE_ONLY, *_AUDIO_ONLY)
        return _compile_tagger(task, ctx, hp)
    if task.id in tasks.IMAGE_TASKS:
        _refuse(hp, task, *_TABULAR_ONLY, *_AUDIO_ONLY, *_TEXT_ONLY)
        return _compile_image(task, ctx, hp, custom)
    if task.id in tasks.AUDIO_TASKS:
        _refuse(hp, task, *_TABULAR_ONLY, *_IMAGE_ONLY, *_TEXT_ONLY)
        return _compile_audio(task, ctx, hp)
    if task.id in tasks.TEXT_TASKS:
        _refuse(hp, task, *_IMAGE_ONLY, *_AUDIO_ONLY)
        return _compile_text(task, ctx, hp, custom)
    _refuse(hp, task, *_IMAGE_ONLY, *_PRETRAINED, *_AUDIO_ONLY, *_TEXT_ONLY)
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


def _compile_image(
    task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters, custom: CustomModelRef | None
) -> dict[str, Any]:
    if len(ctx.label_class_names) < 2:
        raise ConfigError("Image classification needs a snapshot with at least two classes")
    if custom is not None:
        # The model dictates its own image size and normalisation (see custom.py), and the files are not necessarily
        # on disk yet, so `train.py` reads both from them when training starts and writes them into names.json. What
        # is here is a stand-in, and `imageSize` is ignored (the form always sends it).
        if hp.pretrained is False:
            raise ConfigError(
                "A bring-your-own model always starts from its own weights, so it cannot be trained from scratch"
            )
        model_id, size, pretrained = hp.model_id or f"custom:{custom.id}", tasks.IMAGE_MODELS[0].size, True
    else:
        model_id = hp.model_id or tasks.IMAGE_MODELS[0].id
        model = tasks.MODELS_BY_ID.get(model_id)
        if model is None:
            raise ConfigError(f'"{model_id}" is not an image model for the task "{task.id}"')
        size = hp.image_size or model.size
        if model.fixed_size and size != model.size:
            raise ConfigError(f"{model.label} only works at {model.size} pixels, not {size}")
        pretrained = True if hp.pretrained is None else hp.pretrained

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
        "pretrained": pretrained,
        # Where the bring-your-own model's files are, for train.py. Absent for a built-in model.
        **({"custom": {"kind": custom.kind, "path": custom.local_path}} if custom is not None else {}),
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


def _compile_text(
    task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters, custom: CustomModelRef | None
) -> dict[str, Any]:
    if len(ctx.label_class_names) < 2:
        raise ConfigError("Text classification needs a snapshot with at least two classes")
    if custom is not None:
        return _compile_hf_text(ctx, hp, custom)
    model_id = hp.model_id or tasks.TEXT_MODELS[0].id
    if model_id not in tasks.TEXT_MODELS_BY_ID:
        raise ConfigError(f'"{model_id}" is not a text model for the task "{task.id}"')
    if hp.freeze_backbone or hp.pretrained:
        # The form always offers the freeze switch (a bring-your-own model may use it), so only turning it ON is wrong.
        raise ConfigError("The built-in text models are trained from scratch, so there is nothing pretrained to freeze")

    d = tasks.TEXT_DEFAULTS
    # The vocabulary is empty here: `train.py` builds it from the train texts and writes the real one to names.json.
    spec = TextSpec(max_tokens=hp.max_tokens or d["max_tokens"])
    return {
        "backend": "pytorch",
        "inputKind": "text",
        "task": "classify",
        "model": model_id,
        "textColumn": _column(ctx, "inline_text"),
        "labelColumn": _column(ctx, "label"),
        # The index a class gets is its position here. `train.py` writes this order into names.json, and a loaded
        # model reads it back from there: serving and export never consult this config or the label_classes table.
        "classNames": list(ctx.label_class_names),
        "text": {
            "maxTokens": spec.max_tokens,
            "vocabSize": hp.vocab_size or d["vocab_size"],
            "minWordCount": hp.min_word_count or d["min_word_count"],
        },
        "network": {"dropout": d["dropout"] if hp.dropout is None else hp.dropout},
        "train": {
            **_train_block(hp, d, epochs=hp.epochs or d["epochs"]),
            "useClassWeights": bool(hp.use_class_weights),
        },
    }


def _compile_tagger(task: TaskDescriptor, ctx: SnapshotContext, hp: PytorchHyperparameters) -> dict[str, Any]:
    """Token classification with the built-in tagger. The tag set is not known here (a snapshot of tags has no class
    list): `train.py` reads it off the usable rows and writes it, in the model's own order, to names.json."""
    model_id = hp.model_id or tasks.TAGGER_MODELS[0].id
    if model_id not in tasks.TAGGER_MODELS_BY_ID:
        raise ConfigError(f'"{model_id}" is not a token classification model for the task "{task.id}"')
    if hp.freeze_backbone or hp.pretrained:
        raise ConfigError("The built-in tagger is trained from scratch, so there is nothing pretrained to freeze")

    d = tasks.TEXT_DEFAULTS
    return {
        "backend": "pytorch",
        "inputKind": "words",
        "task": "tag",
        "model": model_id,
        "textColumn": _column(ctx, "inline_text"),
        "labelColumn": _column(ctx, "label"),
        "classNames": [],
        "text": {
            "maxTokens": hp.max_tokens or d["max_tokens"],
            "vocabSize": hp.vocab_size or d["vocab_size"],
            "minWordCount": hp.min_word_count or d["min_word_count"],
        },
        "network": {"dropout": d["dropout"] if hp.dropout is None else hp.dropout},
        "train": {
            **_train_block(hp, d, epochs=hp.epochs or d["epochs"]),
            "useClassWeights": bool(hp.use_class_weights),
        },
    }


def _compile_hf_text(ctx: SnapshotContext, hp: PytorchHyperparameters, custom: CustomModelRef) -> dict[str, Any]:
    """A bring-your-own transformer. It brings its own tokenizer and vocabulary, so those settings (and the dropout the
    built-in models have) are ignored (the form always sends them), and it is fine-tuned, so the learning rate defaults
    to a much smaller one. How many tokens it may read depends on its files, which are not necessarily on disk yet, so
    `train.py` checks `maxTokens` against them."""
    if hp.pretrained is False:
        raise ConfigError(
            "A bring-your-own model always starts from its own weights, so it cannot be trained from scratch"
        )
    d = {**tasks.TEXT_DEFAULTS, "learning_rate": tasks.HF_TEXT_LEARNING_RATE}
    return {
        "backend": "pytorch",
        "inputKind": "text",
        "task": "classify",
        "model": hp.model_id or f"custom:{custom.id}",
        "pretrained": True,
        # Where the bring-your-own model's files are, for train.py.
        "custom": {"kind": custom.kind, "path": custom.local_path},
        "textColumn": _column(ctx, "inline_text"),
        "labelColumn": _column(ctx, "label"),
        # The index a class gets is its position here. `train.py` writes this order into names.json, and a loaded
        # model reads it back from there: serving and export never consult this config or the label_classes table.
        "classNames": list(ctx.label_class_names),
        "text": {"maxTokens": hp.max_tokens or d["max_tokens"]},
        "train": {
            **_train_block(hp, d, epochs=hp.epochs or d["epochs"]),
            "freezeBackbone": bool(hp.freeze_backbone),
            "useClassWeights": bool(hp.use_class_weights),
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
    if task.id in tasks.TEXT_TASKS or task.id in tasks.TOKEN_TASKS:
        return _text_specs()
    return []


def _text_specs() -> list[ParamSpec]:
    d = tasks.TEXT_DEFAULTS
    return [
        ParamSpec(
            name="freezeBackbone", label="Freeze backbone", type="bool", group=_MODEL, default=False,
            description="For a bring-your-own Hugging Face model: trains only the final classification layer and "
            "leaves the rest exactly as pretrained. Much faster and needs very few texts, but it adapts less closely "
            "to your data. The built-in models are trained from scratch and have nothing to freeze.",
        ),
        ParamSpec(
            name="epochs", label="Epochs", type="int", group=_TRAINING, default=d["epochs"], min=1, max=500, step=1,
            description="How many times the model goes through your whole training set (each pass is one epoch).",
        ),
        ParamSpec(
            name="batchSize", label="Batch Size", type="choice", group=_TRAINING,
            default=str(d["batch"]), choices=[str(n) for n in tasks.TEXT_BATCH_SIZES],
            description="How many texts the model reads before it updates itself. Bigger batches are faster but "
            "need more memory; lower this first if a Hugging Face model runs out of memory.",
        ),
        ParamSpec(
            name="learningRate", label="Learning Rate", type="float", group=_TRAINING,
            default=None, min=0.00001, max=0.1, step=0.00001,
            description="How big a step the model takes each time it learns from some texts. Too high and it "
            "overshoots; too low and training is slow. Left unset it is "
            f"{d['learning_rate']:g} for the built-in models, which learn from scratch, and "
            f"{tasks.HF_TEXT_LEARNING_RATE:g} for a Hugging Face model, whose pretrained weights a larger step would "
            "wreck.",
        ),
        seed_spec(_TRAINING),
        weight_decay_spec(_TRAINING),
        lr_schedule_spec(_TRAINING),
        warmup_spec(_TRAINING),
        grad_clip_spec(_TRAINING),
        mixed_precision_spec(_TRAINING),
        _optimizer_spec(),
        ParamSpec(
            name="useClassWeights", label="Balance classes", type="bool", group=_TRAINING, default=False,
            description="Weights each class inversely to how often it appears, so a rare class counts as much as a "
            "common one. Helps when one class is far more common than the others, at the cost of some accuracy on "
            "the common one.",
        ),
        ParamSpec(
            name="dropout", label="Dropout", type="float", group=_MODEL, default=d["dropout"], min=0, max=0.9,
            step=0.05,
            description="Randomly switches off this share of the model's units while training, so it cannot lean on "
            "any one of them. Raise it if the model memorises the training texts (validation results get worse "
            "while training results keep improving), which is common with few texts. A Hugging Face model keeps its "
            "own dropout and ignores this.",
        ),
        _patience_spec(d["patience"]),
        ParamSpec(
            name="maxTokens", label="Words read per text", type="int", group=_TEXT, default=d["max_tokens"], min=8,
            max=1024, step=8,
            description="The model reads this many words (for a Hugging Face model, tokens: pieces of words) from the "
            "start of each text and ignores the rest. Texts shorter than this are padded. Reading more of a long text "
            "costs more time and memory, and a Hugging Face model has a limit of its own, usually 512.",
        ),
        ParamSpec(
            name="vocabSize", label="Vocabulary size", type="int", group=_TEXT, default=d["vocab_size"], min=100,
            max=200000, step=1000,
            description="The model learns at most this many different words, the most frequent in your training "
            "texts. Every other word is read as 'unknown'. A bigger vocabulary makes a bigger exported model. A "
            "Hugging Face model brings its own vocabulary and ignores this.",
        ),
        ParamSpec(
            name="minWordCount", label="Minimum word count", type="int", group=_TEXT, default=d["min_word_count"],
            min=1, max=1000, step=1,
            description="A word must appear at least this many times in the training texts to be learned. Raising "
            "it drops typos and one-off names, which the model could only memorise. A Hugging Face model ignores this.",
        ),
    ]  # fmt: skip


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
            "scratch. A bring-your-own model always starts from its own weights.",
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
            "transformer only works at 224. A Hugging Face model uses its own size and ignores this.",
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
