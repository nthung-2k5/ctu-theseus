"""Per-task Ultralytics tables: which YOLO models a task offers and the training defaults.

Pure data: no ultralytics/torch import, so listing this backend never needs a GPU runtime.
"""

from dataclasses import dataclass
from typing import Literal

# Ultralytics' own task string, which picks the trainer/validator/predictor.
YoloTaskKind = Literal["classify", "detect"]


@dataclass(frozen=True)
class YoloModel:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class YoloTaskSpec:
    kind: YoloTaskKind
    models: tuple[YoloModel, ...]
    default_image_size: int
    default_epochs: int
    default_batch_size: int


_CLASSIFY_MODELS = (
    YoloModel("yolo11n-cls", "YOLO11 Nano", "Smallest and fastest (about 1.6M parameters)."),
    YoloModel("yolo11s-cls", "YOLO11 Small", "A bit more accurate than Nano (about 5.5M parameters)."),
    YoloModel("yolo11m-cls", "YOLO11 Medium", "Balanced accuracy and speed (about 10M parameters)."),
    YoloModel("yolo11l-cls", "YOLO11 Large", "More accurate, slower (about 13M parameters)."),
    YoloModel("yolo11x-cls", "YOLO11 Extra Large", "Most accurate, slowest (about 29M parameters)."),
)

_DETECT_MODELS = (
    YoloModel("yolo11n", "YOLO11 Nano", "Smallest and fastest (about 2.6M parameters)."),
    YoloModel("yolo11s", "YOLO11 Small", "A bit more accurate than Nano (about 9.4M parameters)."),
    YoloModel("yolo11m", "YOLO11 Medium", "Balanced accuracy and speed (about 20M parameters)."),
    YoloModel("yolo11l", "YOLO11 Large", "More accurate, slower (about 25M parameters)."),
    YoloModel("yolo11x", "YOLO11 Extra Large", "Most accurate, slowest (about 57M parameters)."),
)

YOLO_TASKS: dict[str, YoloTaskSpec] = {
    "image_classification": YoloTaskSpec(
        kind="classify",
        models=_CLASSIFY_MODELS,
        default_image_size=224,
        default_epochs=30,
        default_batch_size=16,
    ),
    "object_detection": YoloTaskSpec(
        kind="detect",
        models=_DETECT_MODELS,
        default_image_size=640,
        default_epochs=50,
        default_batch_size=16,
    ),
}

# Square sizes offered in the form. Ultralytics needs a multiple of 32 (its maximum network stride) and
# rounds anything else up, so only these are offered.
IMAGE_SIZES = [224, 320, 416, 512, 640]
OPTIMIZERS = ["AdamW", "Adam", "SGD"]
