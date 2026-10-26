"""Per-task model catalog and defaults for the PyTorch backend.

Pure data: no torch or torchvision import, so listing this backend never needs a GPU runtime. Every image model is a
torchvision classifier (BSD-3-Clause) fine-tuned by swapping in a new classification head. The ids are the ones the
Ludwig backend's image encoders already use, so a sweep or a stored habit that names `resnet50` keeps meaning the same
architecture. A table is learned by a small multilayer perceptron trained from scratch, and a sound by a
convolutional network over its log-mel spectrogram.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageModel:
    id: str
    label: str
    description: str
    # `torchvision.models.<builder>` and the `<weights>.DEFAULT` enum that goes with it.
    builder: str
    weights: str
    # Dotted path of the final linear layer inside the module, which is replaced by one with the right class count.
    head: str
    # Side of the square the pretrained weights were trained on, which is also the default here.
    size: int
    # True for a network whose position embeddings are tied to `size` (a vision transformer): no other size works.
    fixed_size: bool = False


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_MODELS: tuple[ImageModel, ...] = (
    ImageModel(
        "resnet18", "ResNet-18", "Small and quick, a solid first baseline (about 11M parameters).",
        "resnet18", "ResNet18_Weights", "fc", 224,
    ),
    ImageModel(
        "resnet50", "ResNet-50", "A deeper ResNet that is usually noticeably more accurate (about 26M parameters).",
        "resnet50", "ResNet50_Weights", "fc", 224,
    ),
    ImageModel(
        "efficientnet_b0", "EfficientNet-B0", "Accurate for its size, a good default when speed matters (about 5M "
        "parameters).",
        "efficientnet_b0", "EfficientNet_B0_Weights", "classifier.1", 224,
    ),
    ImageModel(
        "mobilenet_v3_small", "MobileNetV3-Small", "The lightest here, built for phones and small devices (about 2.5M "
        "parameters).",
        "mobilenet_v3_small", "MobileNet_V3_Small_Weights", "classifier.3", 224,
    ),
    ImageModel(
        "convnext_tiny", "ConvNeXt-Tiny", "A modern convolutional network that often beats a ResNet-50 (about 28M "
        "parameters).",
        "convnext_tiny", "ConvNeXt_Tiny_Weights", "classifier.2", 224,
    ),
    ImageModel(
        "vit_base", "ViT-Base/16", "A vision transformer: the slowest and hungriest here, and it only works at 224 "
        "pixels (about 86M parameters).",
        "vit_b_16", "ViT_B_16_Weights", "heads.head", 224, fixed_size=True,
    ),
)  # fmt: skip

MODELS_BY_ID: dict[str, ImageModel] = {m.id: m for m in IMAGE_MODELS}

IMAGE_TASKS: frozenset[str] = frozenset({"image_classification"})

IMAGE_SIZES = [64, 128, 160, 224, 256, 288, 384]
OPTIMIZERS = ["AdamW", "SGD"]

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_EARLY_STOP_PATIENCE = 5


@dataclass(frozen=True)
class TabularModel:
    id: str
    label: str
    description: str
    # Width of each hidden layer, in order.
    hidden: tuple[int, ...]


TABULAR_MODELS: tuple[TabularModel, ...] = (
    TabularModel(
        "mlp_medium", "MLP (medium)", "Three hidden layers of 128, 128 and 64 units: a good default for most tables.",
        (128, 128, 64),
    ),
    TabularModel(
        "mlp_small", "MLP (small)", "Two hidden layers of 64 units. Trains fastest and overfits least on a small "
        "table.",
        (64, 64),
    ),
    TabularModel(
        "mlp_large", "MLP (large)", "Four hidden layers of 256, 256, 128 and 64 units, for a big table with many "
        "columns and interactions.",
        (256, 256, 128, 64),
    ),
)  # fmt: skip

TABULAR_MODELS_BY_ID: dict[str, TabularModel] = {m.id: m for m in TABULAR_MODELS}

# Task id -> what one row is predicted as.
TABULAR_TASKS: dict[str, str] = {"tabular_classification": "classify", "tabular_regression": "regress"}

TABULAR_DEFAULTS = {"epochs": 60, "batch": 128, "learning_rate": 0.001, "patience": 10, "dropout": 0.1}
TABULAR_BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]


@dataclass(frozen=True)
class AudioModel:
    id: str
    label: str
    description: str
    # True when it starts from ImageNet weights (a spectrogram is a one-channel image to a ResNet).
    pretrained: bool


AUDIO_MODELS: tuple[AudioModel, ...] = (
    AudioModel(
        "audio_resnet18", "ResNet-18 (pretrained)", "A ResNet-18 that starts from ImageNet weights, which transfer "
        "well to spectrograms. The most accurate choice when you have few recordings.",
        True,
    ),
    AudioModel(
        "audio_cnn", "Small CNN (from scratch)", "Four small convolution blocks trained from nothing. Fast and light, "
        "and fine for clearly different sounds when you have plenty of recordings.",
        False,
    ),
)  # fmt: skip

AUDIO_MODELS_BY_ID: dict[str, AudioModel] = {m.id: m for m in AUDIO_MODELS}

AUDIO_TASKS: frozenset[str] = frozenset({"audio_classification"})

AUDIO_SAMPLE_RATES = [8000, 16000, 22050, 32000, 44100]
AUDIO_MEL_BANDS = [40, 64, 96, 128]
AUDIO_DEFAULTS = {
    "epochs": 30,
    "batch": 16,
    "learning_rate": 0.0005,
    "patience": 6,
    "sample_rate": 16000,
    "clip_seconds": 5.0,
    "n_mels": 64,
    "clips_per_audio": 3,
}
AUDIO_BATCH_SIZES = [4, 8, 16, 32, 64]


@dataclass(frozen=True)
class TextModel:
    id: str
    label: str
    description: str


TEXT_MODELS: tuple[TextModel, ...] = (
    TextModel(
        "text_bag", "Word bag (fastText-style)", "Averages the learned vector of every word in the text and classifies "
        "the average. Very fast, hard to overfit, and a strong baseline when the topic is given away by which words "
        "appear.",
    ),
    TextModel(
        "text_cnn", "Word CNN", "Slides filters over runs of two, three and four words, so it also sees short phrases "
        "and negations (\"not good\"), not only which words occur. Slower than the bag, and the better choice when "
        "word order carries the meaning.",
    ),
)  # fmt: skip

TEXT_MODELS_BY_ID: dict[str, TextModel] = {m.id: m for m in TEXT_MODELS}

TEXT_TASKS: frozenset[str] = frozenset({"text_classification"})

TEXT_BATCH_SIZES = [16, 32, 64, 128, 256]
TEXT_DEFAULTS = {
    "epochs": 30,
    "batch": 64,
    "learning_rate": 0.002,
    "patience": 5,
    "dropout": 0.3,
    "max_tokens": 128,
    "vocab_size": 20000,
    "min_word_count": 2,
}
# A bring-your-own transformer is fine-tuned, not trained from scratch: the learning rate that suits a randomly
# initialised model would wreck its weights, so it has its own (used when the form leaves the rate unset). The other
# defaults, including the epoch limit, work for both, because early stopping ends a fine-tune after a few epochs.
HF_TEXT_LEARNING_RATE = 0.00002

# Token classification: one tag per whitespace-separated word (`domain/token_tags.py`). The built-in tagger is trained
# from scratch on the same word-id idea as the text classifiers, so it shares their defaults; a bring-your-own
# transformer is the pretrained option.
TOKEN_TASKS: frozenset[str] = frozenset({"token_classification"})
TAGGER_MODELS: tuple[TextModel, ...] = (
    TextModel(
        "tagger_cnn", "Word CNN tagger", "Looks up each word's learned vector and reads it together with the four "
        "words on either side of it, through two convolution layers, to tag it. Trained from scratch and quick: a "
        "good baseline for a tag set whose words give it away, such as names after \"Mr.\"; a pretrained "
        "transformer (add one on the Models page) does better with little data.",
    ),
)  # fmt: skip
TAGGER_MODELS_BY_ID: dict[str, TextModel] = {m.id: m for m in TAGGER_MODELS}
# The convolutions' kernel width (an odd number, so a word sits in the middle of its window) and how many there are.
TAGGER_KERNEL = 5
TAGGER_LAYERS = 2
# The embedding width of each text model: the size of a word's learned vector.
TEXT_EMBEDDING = 128
# A convolution filter spans this many words, and there are this many of each size.
TEXT_CNN_WIDTHS = (2, 3, 4)
TEXT_CNN_FILTERS = 100
