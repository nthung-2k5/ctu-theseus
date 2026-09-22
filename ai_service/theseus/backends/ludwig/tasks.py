"""Per-task Ludwig feature tables: input/output feature builders, encoder catalogs and trainer
knob defaults, keyed by task id.

Moved wholesale out of the framework-neutral `services/task_registry.py` when trainer backends
became a plugin system: everything here is Ludwig's own vocabulary (feature types, encoders,
`ecd`/`llm` model types) and has no business in a registry other backends also read.
"""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from theseus.db.enums import ProjectTask
from theseus.services.task_registry import CLASS, IMAGE_PATH

LudwigFeatureType = Literal["image", "text", "audio", "number", "category", "binary", "sequence", "vector"]

# A common, dependency-free subset of Ludwig optimizers (none need bitsandbytes).
LUDWIG_OPTIMIZER_TYPES = ("adam", "adamw", "sgd", "rmsprop", "adagrad")


class EncoderChoice(BaseModel):
    id: str
    label: str
    # Ludwig encoder type value emitted into the compiled config.
    encoder_type: str
    pretrained: bool
    # Extra Ludwig encoder params merged in verbatim (e.g. torchvision model_variant).
    params: dict[str, Any] | None = None


class IntKnob(BaseModel):
    default: int
    min: int
    max: int | None = None


class FloatKnob(BaseModel):
    default: float
    min: float
    max: float | None = None


class BatchSizeKnob(BaseModel):
    default: int | Literal["auto"]
    options: list[int | Literal["auto"]]


class TrainerKnobSpec(BaseModel):
    epochs: IntKnob
    batch_size: BatchSizeKnob
    learning_rate: FloatKnob
    early_stop_patience: IntKnob


@dataclass
class LudwigTaskSpec:
    model_type: Literal["ecd", "llm"]
    input_features: list[dict[str, Any]]
    output_features: list[dict[str, Any]]
    encoders: list[EncoderChoice]
    trainer_knobs: TrainerKnobSpec


def _feat(name: str, ftype: LudwigFeatureType) -> dict[str, Any]:
    return {"name": name, "type": ftype, "column": name}


def _default_knobs() -> TrainerKnobSpec:
    return TrainerKnobSpec(
        epochs=IntKnob(default=20, min=1, max=500),
        batch_size=BatchSizeKnob(default="auto", options=[16, 32, 64, 128, 256, "auto"]),
        learning_rate=FloatKnob(default=0.001, min=0.00001, max=1),
        early_stop_patience=IntKnob(default=5, min=-1),
    )


def _llm_knobs() -> TrainerKnobSpec:
    return TrainerKnobSpec(
        epochs=IntKnob(default=3, min=1, max=50),
        batch_size=BatchSizeKnob(default=1, options=[1, 2, 4, 8, "auto"]),
        learning_rate=FloatKnob(default=0.0001, min=0.000001, max=0.01),
        early_stop_patience=IntKnob(default=3, min=-1),
    )


# -- Encoder catalogs ------------------------------------------------------------------------

VISION_ENCODERS = [
    EncoderChoice(
        id="resnet18", label="ResNet-18", encoder_type="resnet", pretrained=True, params={"model_variant": 18}
    ),
    EncoderChoice(
        id="resnet50", label="ResNet-50", encoder_type="resnet", pretrained=True, params={"model_variant": 50}
    ),
    EncoderChoice(
        id="vit_base",
        label="ViT-Base/16",
        encoder_type="vit",
        pretrained=True,
        params={"model_variant": "base_patch16_224"},
    ),
    EncoderChoice(
        id="convnext_tiny",
        label="ConvNeXt-Tiny",
        encoder_type="convnext",
        pretrained=True,
        params={"model_variant": "tiny"},
    ),
    EncoderChoice(
        id="efficientnet_b0",
        label="EfficientNet-B0",
        encoder_type="efficientnet",
        pretrained=True,
        params={"model_variant": "b0"},
    ),
    EncoderChoice(
        id="mobilenet_v3_small",
        label="MobileNetV3-Small",
        encoder_type="mobilenetv3",
        pretrained=True,
        params={"model_variant": "small"},
    ),
]

TEXT_ENCODERS = [
    EncoderChoice(id="bert", label="BERT", encoder_type="bert", pretrained=True),
    EncoderChoice(id="distilbert", label="DistilBERT", encoder_type="distilbert", pretrained=True),
    EncoderChoice(id="roberta", label="RoBERTa", encoder_type="roberta", pretrained=True),
    EncoderChoice(
        id="stacked_cnn", label="Stacked CNN (train from scratch)", encoder_type="stacked_cnn", pretrained=False
    ),
]

AUDIO_ENCODERS = [
    EncoderChoice(id="stacked_cnn", label="Stacked CNN", encoder_type="stacked_cnn", pretrained=False),
    EncoderChoice(id="rnn", label="RNN", encoder_type="rnn", pretrained=False),
    EncoderChoice(id="cnnrnn", label="CNN + RNN", encoder_type="cnnrnn", pretrained=False),
]


def _spec(
    *, model_type: Literal["ecd", "llm"] = "ecd", inputs: list[dict[str, Any]], outputs: list[dict[str, Any]],
    encoders: list[EncoderChoice], knobs: TrainerKnobSpec | None = None,
) -> LudwigTaskSpec:  # fmt: skip
    return LudwigTaskSpec(
        model_type=model_type, input_features=inputs, output_features=outputs, encoders=encoders,
        trainer_knobs=knobs or (_llm_knobs() if model_type == "llm" else _default_knobs()),
    )  # fmt: skip


def _llm_text_spec(in_cols: list[str], out_col: str) -> LudwigTaskSpec:
    return _spec(
        model_type="llm", inputs=[_feat(c, "text") for c in in_cols], outputs=[_feat(out_col, "text")], encoders=[]
    )


def _file_caption_spec(
    path_col: str, out_col: str, in_type: LudwigFeatureType, encoders: list[EncoderChoice]
) -> LudwigTaskSpec:
    return _spec(inputs=[_feat(path_col, in_type)], outputs=[_feat(out_col, "text")], encoders=encoders)


# -- Task table --------------------------------------------------------------------------------
# Every task Ludwig supports today (the ones with status != "planned" in task_registry.py).

LUDWIG_TASKS: dict[str, LudwigTaskSpec] = {
    "image_classification": _spec(
        inputs=[_feat(IMAGE_PATH, "image")], outputs=[_feat(CLASS, "category")], encoders=VISION_ENCODERS
    ),
    "text_classification": _spec(
        inputs=[_feat("text", "text")], outputs=[_feat(CLASS, "category")], encoders=TEXT_ENCODERS
    ),
    # Tabular input columns are dataset-defined, so input_features is empty here and the compiler
    # derives one number feature per scalar column of the snapshot.
    "tabular_classification": _spec(inputs=[], outputs=[_feat(CLASS, "category")], encoders=[]),
    "tabular_regression": _spec(inputs=[], outputs=[_feat("target", "number")], encoders=[]),
    "audio_classification": _spec(
        inputs=[_feat("audio_path", "audio")], outputs=[_feat(CLASS, "category")], encoders=AUDIO_ENCODERS
    ),
    "token_classification": _spec(
        inputs=[_feat("text", "sequence")], outputs=[_feat("tags", "sequence")], encoders=TEXT_ENCODERS
    ),
    "text_generation": _llm_text_spec(["prompt"], "completion"),
    "summarization": _llm_text_spec(["document"], "summary"),
    "sequence_to_sequence": _llm_text_spec(["source"], "target"),
    "question_answering": _llm_text_spec(["context", "question"], "answer"),
    "image_captioning": _file_caption_spec(IMAGE_PATH, "caption", "image", VISION_ENCODERS),
    "audio_captioning": _file_caption_spec("audio_path", "caption", "audio", AUDIO_ENCODERS),
    "automatic_speech_recognition": _file_caption_spec("audio_path", "transcript", "audio", AUDIO_ENCODERS),
}

# Sanity: every key above is a real project task.
assert set(LUDWIG_TASKS) <= set(ProjectTask.__args__)  # type: ignore[attr-defined]
