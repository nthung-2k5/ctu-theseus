"""Per-task video model catalog and the clip-sampling defaults.

Pure data: no torch or torchvision import, so listing this backend never needs a GPU runtime. Every model is one of
torchvision's Kinetics-400 video classifiers (BSD-3-Clause), fine-tuned by swapping in a new classification head.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoModel:
    id: str
    label: str
    description: str
    # The side of the square crop the Kinetics weights were trained on, which is also the default here.
    crop: int
    # Frames the architecture was trained on per clip. Others work (the nets are convolutional) but this is the default.
    frames: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD = (0.22803, 0.22145, 0.216989)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

VIDEO_MODELS: tuple[VideoModel, ...] = (
    VideoModel(
        "mc3_18", "MC3-18", "Smallest and fastest: a 3D ResNet mixing 3D and 2D convolutions (about 12M parameters).",
        112, 16, _KINETICS_MEAN, _KINETICS_STD,
    ),
    VideoModel(
        "r3d_18", "R3D-18", "A plain 3D ResNet-18, a solid baseline for action recognition (about 33M parameters).",
        112, 16, _KINETICS_MEAN, _KINETICS_STD,
    ),
    VideoModel(
        "r2plus1d_18", "R(2+1)D-18", "Splits each 3D convolution into space and time, usually a little more accurate "
        "than R3D (about 32M parameters).",
        112, 16, _KINETICS_MEAN, _KINETICS_STD,
    ),
    VideoModel(
        "s3d", "S3D", "A separable 3D network that works at 224 pixels and is light for its accuracy (about 8M "
        "parameters).",
        224, 16, _KINETICS_MEAN, _KINETICS_STD,
    ),
    VideoModel(
        "swin3d_t", "Video Swin-T", "A video transformer, the most accurate and the slowest here (about 28M "
        "parameters).",
        224, 32, _IMAGENET_MEAN, _IMAGENET_STD,
    ),
)  # fmt: skip

MODELS_BY_ID: dict[str, VideoModel] = {m.id: m for m in VIDEO_MODELS}

VIDEO_TASKS: frozenset[str] = frozenset({"video_classification"})

FRAME_COUNTS = [8, 16, 32]
FRAME_SIZES = [64, 112, 160, 224]
OPTIMIZERS = ["AdamW", "SGD"]

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_FRAME_STRIDE = 2
DEFAULT_CLIPS_PER_VIDEO = 3
