"""Bring-your-own video models: Hugging Face VideoMAE, TimeSformer and ViViT classifiers.

This module only reads JSON, so listing the kinds and validating a model's files never needs torch. Building and
running the network is `hf.py`, imported only where a model is really loaded.

What a clip looks like is the model's to decide, not the user's: these networks have position embeddings tied to the
exact number of frames and the exact image size in their `config.json`, and were trained on the mean and standard
deviation in `preprocessor_config.json`. `clip_settings` reads both, and training uses them in place of the
frames/size/normalisation of a built-in model.
"""

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from omnitrain.backends.base import ConfigError, CustomModelKind, CustomModelRef
from omnitrain.services.task_registry import TaskDescriptor

HF_VIDEO = "hf_video"
SUPPORTED_MODEL_TYPES = ("videomae", "timesformer", "vivit")
# The `architectures` entry of a checkpoint that can be fine-tuned into a video classifier: an already-trained
# classifier, a pretraining checkpoint (the usual way a base model is published) or the bare backbone.
_ARCHITECTURE = re.compile(r"^(VideoMAE|Timesformer|Vivit)(For(VideoClassification|PreTraining)|Model)$")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _transformers_unavailable() -> str | None:
    if importlib.util.find_spec("transformers") is None:
        return "The 'transformers' package is not installed on this server"
    return None


def custom_kinds_for(task_id: str) -> list[CustomModelKind]:
    if task_id != "video_classification":
        return []
    return [
        CustomModelKind(
            id=HF_VIDEO,
            label="Hugging Face video model",
            description="A VideoMAE, TimeSformer or ViViT checkpoint (config.json plus .safetensors weights). "
            "Its own frame count, image size and normalisation are used, so the clip settings above are ignored.",
            modality="video",
            source_kinds=["hub", "upload"],
            status="experimental",
            unavailable_reason=_transformers_unavailable(),
        )
    ]


def suggest_kind(config: dict[str, Any]) -> str | None:
    architectures = [a for a in config.get("architectures") or [] if isinstance(a, str)]
    if config.get("model_type") in SUPPORTED_MODEL_TYPES and any(_ARCHITECTURE.match(a) for a in architectures):
        return HF_VIDEO
    return None


def read_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "config.json"
    if not path.is_file():
        raise ConfigError("The model has no config.json at its top level (expected a Hugging Face model folder)")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"config.json could not be read: {e}") from None
    if not isinstance(config, dict):
        raise ConfigError("config.json is not a JSON object")
    return config


def clip_settings(root: str | Path) -> dict[str, Any]:
    """The clip a model was built for: `numFrames`, `frameSize`, `mean` and `std`."""
    config = read_config(root)
    frames, size = config.get("num_frames"), config.get("image_size")
    if not isinstance(frames, int) or not isinstance(size, int) or frames < 2 or size < 16:
        raise ConfigError("config.json does not say how many frames and what image size the model takes")
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    preprocessor = Path(root) / "preprocessor_config.json"
    if preprocessor.is_file():
        try:
            found = json.loads(preprocessor.read_text(encoding="utf-8"))
            if _triple(found.get("image_mean")) and _triple(found.get("image_std")):
                mean, std = list(found["image_mean"]), list(found["image_std"])
        except (OSError, ValueError, AttributeError):
            pass  # unreadable: the ImageNet statistics, which most video checkpoints use
    return {"numFrames": frames, "frameSize": size, "mean": [float(v) for v in mean], "std": [float(v) for v in std]}


def _triple(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value)


def validate_custom_model(task: TaskDescriptor, ref: CustomModelRef) -> None:
    if ref.kind not in {k.id for k in custom_kinds_for(task.id)}:
        raise ConfigError(f'A "{ref.kind}" model cannot be used for {task.label}')
    root = Path(ref.local_path)
    config = read_config(root)
    if "auto_map" in config:
        # auto_map makes transformers import and run Python from the model repo. Never do that with a file a
        # user (or a Hub author) supplied.
        raise ConfigError("Models that need custom code (auto_map / trust_remote_code) are not supported")
    if config.get("model_type") not in SUPPORTED_MODEL_TYPES:
        listed = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ConfigError(f'Not a supported video model (model_type "{config.get("model_type")}"; expected {listed})')
    if not any(root.glob("*.safetensors")):
        raise ConfigError("The model has no .safetensors weights (pickle-based weights are not accepted)")
    clip_settings(root)

    # Prove it works: build the very network training will use and run one blank clip through it. That catches
    # what a config check cannot, such as weights that do not match the config.
    from omnitrain.backends.video import hf

    try:
        hf.smoke_test(str(root))
    except Exception as e:  # noqa: BLE001  any failure here means the model cannot be fine-tuned as a classifier
        text = str(e).strip() or type(e).__name__
        raise ConfigError(f"This model cannot be loaded as a video classifier: {text.splitlines()[0][:300]}") from None
