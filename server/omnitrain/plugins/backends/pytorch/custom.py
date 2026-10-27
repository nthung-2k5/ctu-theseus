"""Bring-your-own models: Hugging Face `transformers` image classifiers (ViT, Swin, ConvNeXt, ResNet, ...), text
classifiers and token classifiers (BERT and its relatives).

This module only reads JSON, so listing the kinds and validating a model's files never needs torch. Building and
running the network is `hf.py`, imported only where a model is really loaded.

What an image looks like is the model's to decide, not the user's: a vision transformer's position embeddings are tied
to the size in its `config.json`, and every checkpoint was trained on the mean and standard deviation in its
`preprocessor_config.json`. `image_settings` reads both, and training uses them in place of the image size and
ImageNet statistics of a built-in model. The recipe around them stays this backend's own (`images.preprocess`: the
shorter edge to that size, a centre crop, `(x - mean) / std`), which is what every generated client already implements,
so a bring-your-own image model exports and serves exactly like a built-in one.

A text is the opposite: the recipe is the model's own tokenizer (`tokenizer.json`, run by Hugging Face `tokenizers`),
and the only thing this backend decides is how many tokens are read, within what the model's position embeddings
allow (`text_token_limit`).
"""

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from omnitrain.domain.task_registry import TaskDescriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelKind, CustomModelRef
from omnitrain.plugins.backends.pytorch import tasks

HF_IMAGE = "hf_image"
HF_TEXT = "hf_text"
HF_TOKEN = "hf_token"
# The kinds that read a text with the model's own tokenizer.
HF_TEXT_KINDS = (HF_TEXT, HF_TOKEN)
# Families whose classifier loads, trains and exports to ONNX (`tests/test_pytorch_custom.py` runs each through all
# three on a tiny random model). Another family is one more entry here and one more row in that test.
SUPPORTED_MODEL_TYPES = ("vit", "deit", "beit", "swin", "swinv2", "convnext", "convnextv2", "resnet", "mobilenet_v2")
SUPPORTED_TEXT_MODEL_TYPES = ("bert", "distilbert", "roberta", "xlm-roberta", "albert", "electra")
# The text families whose position ids start after the padding id, so the last `pad_token_id + 1` positions cannot be
# used (RoBERTa's 514 positions hold 512 tokens).
_POSITIONS_AFTER_PAD = ("roberta", "xlm-roberta")
_TEXT_ARCHITECTURE = re.compile(
    r"^(Bert|DistilBert|Roberta|XLMRoberta|Albert|Electra)"
    r"(ForSequenceClassification|ForMaskedLM|ForPreTraining|Model)$"
)
# A checkpoint that is already a tagger (or a bare encoder, matched above) can be fine-tuned into one.
_TOKEN_ARCHITECTURE = re.compile(r"^(Bert|DistilBert|Roberta|XLMRoberta|Albert|Electra)ForTokenClassification$")
# The `architectures` entry of a checkpoint that can be fine-tuned into an image classifier: an already-trained
# classifier or the bare backbone (the usual way a base model is published). `DeiTForImageClassificationWithTeacher`,
# whose two heads are averaged, is deliberately not matched.
_ARCHITECTURE = re.compile(
    r"^(ViT|DeiT|Beit|Swin|Swinv2|ConvNext|ConvNextV2|ResNet|MobileNetV2)(ForImageClassification|Model)$"
)

IMAGENET_MEAN = tasks.IMAGENET_MEAN
IMAGENET_STD = tasks.IMAGENET_STD
# What a convolutional network with no size of its own is fed when nothing says otherwise.
DEFAULT_SIZE = 224
MIN_SIZE = 16


def _transformers_unavailable() -> str | None:
    if importlib.util.find_spec("transformers") is None:
        return "The 'transformers' package is not installed on this server"
    return None


def custom_kinds_for(task_id: str) -> list[CustomModelKind]:
    if task_id in tasks.IMAGE_TASKS:
        return [
            CustomModelKind(
                id=HF_IMAGE,
                label="Hugging Face image model",
                description="A ViT, DeiT, BEiT, Swin, ConvNeXt, ResNet or MobileNetV2 image classifier or backbone "
                "(config.json plus .safetensors weights). Its own image size and normalisation are used, so the "
                "image size setting is ignored.",
                modality="vision",
                source_kinds=["hub", "upload"],
                status="experimental",
                unavailable_reason=_transformers_unavailable(),
            )
        ]
    if task_id in tasks.TEXT_TASKS:
        return [
            CustomModelKind(
                id=HF_TEXT,
                label="Hugging Face text model",
                description="A BERT, DistilBERT, RoBERTa, XLM-RoBERTa, ALBERT or ELECTRA encoder (config.json, "
                ".safetensors weights and the tokenizer files, in a form the fast tokenizer reads). It is fine-tuned "
                "with its own tokenizer, so the vocabulary settings are ignored and the learning rate defaults to "
                "0.00002.",
                modality="text",
                source_kinds=["hub", "upload"],
                status="experimental",
                unavailable_reason=_transformers_unavailable(),
            )
        ]
    if task_id in tasks.TOKEN_TASKS:
        return [
            CustomModelKind(
                id=HF_TOKEN,
                label="Hugging Face token classification model",
                description="A BERT, DistilBERT, RoBERTa, XLM-RoBERTa, ALBERT or ELECTRA encoder (config.json, "
                ".safetensors weights and the tokenizer files, in a form the fast tokenizer reads). It is fine-tuned "
                "with its own tokenizer, which may cut a word into several pieces: the first piece of a word carries "
                "that word's tag. The vocabulary settings are ignored and the learning rate defaults to 0.00002.",
                modality="text",
                source_kinds=["hub", "upload"],
                status="experimental",
                unavailable_reason=_transformers_unavailable(),
            )
        ]
    return []


def suggest_kind(config: dict[str, Any]) -> str | None:
    """The kind a checkpoint's config points at. A text encoder is one family for two tasks (the same BERT is a text
    classifier or a tagger once it has a head), so a bare encoder is suggested as a text classifier, the commoner use,
    and a checkpoint that already is a tagger is suggested as one."""
    architectures = [a for a in config.get("architectures") or [] if isinstance(a, str)]
    if config.get("model_type") in SUPPORTED_MODEL_TYPES and any(_ARCHITECTURE.match(a) for a in architectures):
        return HF_IMAGE
    if config.get("model_type") in SUPPORTED_TEXT_MODEL_TYPES:
        if any(_TOKEN_ARCHITECTURE.match(a) for a in architectures):
            return HF_TOKEN
        if any(_TEXT_ARCHITECTURE.match(a) for a in architectures):
            return HF_TEXT
    return None


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{what} could not be read: {e}") from None
    if not isinstance(found, dict):
        raise ConfigError(f"{what} is not a JSON object")
    return found


def read_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "config.json"
    if not path.is_file():
        raise ConfigError("The model has no config.json at its top level (expected a Hugging Face model folder)")
    return read_json(path, "config.json")


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _preprocessor_size(preprocessor: dict[str, Any]) -> int | None:
    """The side of the image a preprocessor feeds the network: its crop if it has one, else its resize target."""
    for key in ("crop_size", "size"):
        value = preprocessor.get(key)
        if _int(value):
            return int(value)
        if isinstance(value, dict):
            for side in ("height", "shortest_edge", "width"):
                if _int(value.get(side)):
                    return int(value[side])
    return None


def _triple(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value)


def image_settings(root: str | Path) -> dict[str, Any]:
    """The image a model was built for: `imageSize`, `mean` and `std`.

    The size is the config's `image_size` when it has one (a transformer's position embeddings, or a classifier's
    final feature map, are tied to it), otherwise what the preprocessor feeds the network, otherwise 224. The
    normalisation is the preprocessor's, otherwise ImageNet's, which most image checkpoints use."""
    config = read_config(root)
    channels = config.get("num_channels", 3)
    if channels != 3:
        raise ConfigError(f"The model takes {channels}-channel images; only RGB (3 channels) is supported")
    preprocessor: dict[str, Any] = {}
    path = Path(root) / "preprocessor_config.json"
    if path.is_file():
        try:
            preprocessor = read_json(path, "preprocessor_config.json")
        except ConfigError:
            preprocessor = {}  # unreadable: ImageNet statistics and the config's size, which most checkpoints use
    if preprocessor.get("do_rescale") is False:
        raise ConfigError("The model expects pixel values that are not scaled to 0..1, which is not supported")

    size = _int(config.get("image_size")) or _preprocessor_size(preprocessor) or DEFAULT_SIZE
    if size < MIN_SIZE:
        raise ConfigError(f"The model's image size ({size}) is smaller than the {MIN_SIZE} pixels this backend allows")
    mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)
    if preprocessor.get("do_normalize") is False:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    elif _triple(preprocessor.get("image_mean")) and _triple(preprocessor.get("image_std")):
        mean, std = list(preprocessor["image_mean"]), list(preprocessor["image_std"])
    if any(float(v) <= 0 for v in std):
        raise ConfigError("The model's normalisation has a standard deviation that is not positive")
    return {"imageSize": size, "mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def text_token_limit(root: str | Path) -> int:
    """The most tokens a text may be cut to for this model: its position embeddings, less the positions a RoBERTa-style
    model leaves unused. Training refuses a longer `maxTokens` (the model would fail in the middle of an epoch)."""
    config = read_config(root)
    positions = _int(config.get("max_position_embeddings"))
    if positions is None:
        raise ConfigError("config.json does not say how many positions the model reads (max_position_embeddings)")
    if config.get("model_type") in _POSITIONS_AFTER_PAD:
        positions -= int(config.get("pad_token_id") or 0) + 1
    if positions < 8:
        raise ConfigError("The model reads too few positions to classify a text")
    return positions


def validate_custom_model(task: TaskDescriptor, ref: CustomModelRef) -> None:
    if ref.kind not in {k.id for k in custom_kinds_for(task.id)}:
        raise ConfigError(f'A "{ref.kind}" model cannot be used for {task.label}')
    root = Path(ref.local_path)
    config = read_config(root)
    if "auto_map" in config:
        # auto_map makes transformers import and run Python from the model repo. Never do that with a file a
        # user (or a Hub author) supplied.
        raise ConfigError("Models that need custom code (auto_map / trust_remote_code) are not supported")
    text = ref.kind in HF_TEXT_KINDS
    families = SUPPORTED_TEXT_MODEL_TYPES if text else SUPPORTED_MODEL_TYPES
    if config.get("model_type") not in families:
        listed = ", ".join(families)
        raise ConfigError(
            f'Not a supported {"text" if text else "image"} model (model_type "{config.get("model_type")}"; '
            f"expected {listed})"
        )
    if not any(root.glob("*.safetensors")):
        raise ConfigError("The model has no .safetensors weights (pickle-based weights are not accepted)")
    if text:
        text_token_limit(root)
    else:
        image_settings(root)

    # Prove it works: build the very network training will use and run one blank image (or one short text) through
    # it. That catches what a config check cannot, such as weights that do not match the config, or a tokenizer that
    # cannot be read.
    from omnitrain.plugins.backends.pytorch import hf

    noun = {HF_TEXT: "a text classifier", HF_TOKEN: "a token classifier"}.get(ref.kind, "an image classifier")
    try:
        if ref.kind == HF_TOKEN:
            hf.smoke_test_tokens(str(root))
        elif text:
            hf.smoke_test_text(str(root))
        else:
            hf.smoke_test(str(root))
    except Exception as e:  # noqa: BLE001  any failure here means the model cannot be fine-tuned as a classifier
        reason = str(e).strip() or type(e).__name__
        raise ConfigError(f"This model cannot be loaded as {noun}: {reason.splitlines()[0][:300]}") from None
