"""The generated Python client must preprocess an image exactly the way the model's framework does.

`imageResize.mode == "shorter_edge_center_crop"` in preprocessing.json (written by the YOLO backend) asks a client
to resize the shorter edge and centre-crop, instead of squashing the whole image to a square. The reference
here is Ultralytics' own `classify_transforms`, so a drift in either side shows up as a pixel difference.
"""

import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import omnitrain.export

CLIENT_PATH = Path(omnitrain.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"

NORMALIZATION = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
CROP = {"mode": "shorter_edge_center_crop", "interpolation": "bilinear"}


@pytest.fixture(scope="module")
def client_module():
    # Compiled from source rather than imported: an import would write a __pycache__ next to the template, and
    # a .pyc is not UTF-8, which the template-shipping tests (and a devkit bundle) must never see there.
    module = types.ModuleType("omnitrain_client_under_test")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    return module


def make_client(module, size, **extra):
    """A client with only the manifest input set: `_preprocess_image` needs nothing else (no ONNX session)."""
    client = object.__new__(module.OmniTrainClient)
    client.input_spec = {
        "type": "image",
        "ludwigPreprocessing": {"height": size, "width": size},
        "imageNormalization": NORMALIZATION,
        **extra,
    }
    return client


def write_image(path, width, height, seed=0):
    pixels = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


# (width, height, size): landscape, portrait, square, odd leftovers (round-half-to-even), tiny sides that are
# upscaled, and extreme aspect ratios.
SHAPES = [
    (100, 60, 32),
    (60, 100, 32),
    (64, 64, 32),
    (301, 224, 64),
    (224, 301, 64),
    (225, 448, 64),
    (33, 500, 32),
    (500, 33, 32),
    (50, 30, 64),
    (31, 97, 64),
    (97, 31, 64),
    (35, 32, 32),  # leftover 3: where round-half-to-even (2) differs from floor (1)
    (32, 35, 32),
]


@pytest.mark.parametrize(("width", "height", "size"), SHAPES)
def test_matches_ultralytics_classify_transforms_pixel_for_pixel(client_module, tmp_path, width, height, size):
    from ultralytics.data.augment import classify_transforms

    path = write_image(tmp_path / "x.png", width, height, seed=width * 1000 + height)
    expected = classify_transforms(size)(Image.open(path).convert("RGB")).numpy()  # CHW float, 0..1

    actual = make_client(client_module, size, imageResize=CROP)._preprocess_image(path)

    assert actual.shape == (1, 3, size, size)
    np.testing.assert_allclose(actual[0], expected, atol=1e-6)


def test_without_an_image_resize_the_whole_image_is_still_squashed(client_module, tmp_path):
    # Ludwig exports carry no imageResize, and their behaviour must not change.
    path = write_image(tmp_path / "x.png", 100, 60)
    expected = np.asarray(Image.open(path).convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0

    actual = make_client(client_module, 32)._preprocess_image(path)

    np.testing.assert_allclose(actual[0], expected.transpose(2, 0, 1), atol=1e-6)


def test_the_crop_and_the_squash_really_differ_on_a_non_square_image(client_module, tmp_path):
    path = write_image(tmp_path / "x.png", 100, 60)
    cropped = make_client(client_module, 32, imageResize=CROP)._preprocess_image(path)
    squashed = make_client(client_module, 32)._preprocess_image(path)
    assert not np.allclose(cropped, squashed, atol=1e-3)


def test_an_unknown_mode_is_refused_not_silently_squashed(client_module, tmp_path):
    path = write_image(tmp_path / "x.png", 40, 40)
    with pytest.raises(NotImplementedError, match="rotate_and_pad"):
        make_client(client_module, 32, imageResize={"mode": "rotate_and_pad"})._preprocess_image(path)


@pytest.mark.parametrize(
    ("width", "height", "size", "expected"),
    [
        # (resized width, resized height, crop left, crop top), worked out by hand.
        (100, 60, 32, (53, 32, 10, 0)),  # 32*100/60 = 53.3 -> 53 (truncated); leftover 21 -> 10.5 -> 10 (half to even)
        (60, 100, 32, (32, 53, 0, 10)),
        (64, 64, 32, (32, 32, 0, 0)),
        (301, 224, 64, (86, 64, 11, 0)),  # 64*301/224 = 86 exactly; leftover 22 -> 11
        (35, 32, 32, (35, 32, 2, 0)),  # leftover 3 -> 1.5 -> 2: half to even rounds UP here, so it is not floor
        (32, 35, 32, (32, 35, 0, 2)),
    ],
)
def test_geometry_is_the_torchvision_arithmetic(client_module, width, height, size, expected):
    assert client_module._shorter_edge_geometry(width, height, size) == expected
