"""Bring-your-own Hugging Face image models on the PyTorch backend: kinds, file checks, compile, strict loading of a
classifier or a bare backbone, and a real tiny train / load / predict / evaluate / ONNX round trip. Every model is a
randomly initialised few-layer network written to a temp folder, so nothing is downloaded."""

import json
import os
import shutil
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("transformers")

import omnitrain.plugins.export  # noqa: E402
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor  # noqa: E402
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, TrainContext  # noqa: E402
from omnitrain.plugins.backends.pytorch import PytorchBackend, custom, data, hf, runtime  # noqa: E402
from omnitrain.plugins.backends.pytorch import train as train_mod  # noqa: E402
from omnitrain.plugins.backends.pytorch.compile import PytorchHyperparameters, compile_pytorch_config  # noqa: E402
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest  # noqa: E402
from omnitrain.plugins.backends.pytorch.model import ImageModel, PytorchLoadedModel, wrap  # noqa: E402
from omnitrain.plugins.backends.registry import get_backend  # noqa: E402

TASK = get_task_descriptor("image_classification")
CLASSES = ["red", "green", "blue"]
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="image_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=CLASSES,
)
CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"

# A tiny config per family: (config class, its arguments, bare backbone class, classifier class), all by name.
_VIT = {"image_size": 32, "patch_size": 8, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
        "intermediate_size": 64}  # fmt: skip
FAMILIES = {
    "vit": ("ViTConfig", _VIT, "ViTModel", "ViTForImageClassification"),
    "deit": ("DeiTConfig", _VIT, "DeiTModel", "DeiTForImageClassification"),
    "beit": ("BeitConfig", _VIT, "BeitModel", "BeitForImageClassification"),
    "swin": (
        "SwinConfig",
        {"image_size": 32, "patch_size": 4, "embed_dim": 16, "depths": [1, 1], "num_heads": [2, 2], "window_size": 4},
        "SwinModel",
        "SwinForImageClassification",
    ),
    "swinv2": (
        "Swinv2Config",
        {"image_size": 32, "patch_size": 4, "embed_dim": 16, "depths": [1, 1], "num_heads": [2, 2], "window_size": 4},
        "Swinv2Model",
        "Swinv2ForImageClassification",
    ),
    "convnext": (
        "ConvNextConfig",
        {"num_stages": 2, "hidden_sizes": [16, 32], "depths": [1, 1], "image_size": 32},
        "ConvNextModel",
        "ConvNextForImageClassification",
    ),
    "convnextv2": (
        "ConvNextV2Config",
        {"num_stages": 2, "hidden_sizes": [16, 32], "depths": [1, 1], "image_size": 32},
        "ConvNextV2Model",
        "ConvNextV2ForImageClassification",
    ),
    "resnet": (
        "ResNetConfig",
        {"embedding_size": 16, "hidden_sizes": [16, 32], "depths": [1, 1], "layer_type": "basic"},
        "ResNetModel",
        "ResNetForImageClassification",
    ),
    "mobilenet_v2": (
        "MobileNetV2Config",
        {"image_size": 32, "depth_multiplier": 0.25},
        "MobileNetV2Model",
        "MobileNetV2ForImageClassification",
    ),
}


def make_tiny(path: Path, family: str = "vit", *, bare: bool = False, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
    """A random few-layer checkpoint of `family`, as a classifier (5 labels) or as a bare backbone. Returns the model
    that was saved, so a test can compare weights against it."""
    import transformers

    config_name, arguments, backbone_name, classifier_name = FAMILIES[family]
    if bare:
        model = getattr(transformers, backbone_name)(getattr(transformers, config_name)(**arguments))
    else:
        model = getattr(transformers, classifier_name)(getattr(transformers, config_name)(**arguments, num_labels=5))
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    (path / "preprocessor_config.json").write_text(json.dumps({"image_mean": list(mean), "image_std": list(std)}))
    return model


def ref_for(path: Path, kind: str = custom.HF_IMAGE) -> CustomModelRef:
    return CustomModelRef(id="u1", kind=kind, source_kind="upload", local_path=str(path))


def copy_of(source: Path, tmp_path: Path) -> Path:
    folder = tmp_path / "m"
    shutil.copytree(source, folder)
    return folder


def edit_config(folder: Path, **changes) -> None:
    config = json.loads((folder / "config.json").read_text())
    (folder / "config.json").write_text(json.dumps({**config, **changes}))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vit")
    make_tiny(path)
    return path


class TestKindAndValidation:
    def test_the_kind_is_offered_for_image_classification_only(self):
        [kind] = PytorchBackend.custom_model_kinds(TASK)
        assert (kind.id, kind.modality, kind.source_kinds, kind.status) == (
            "hf_image",
            "vision",
            ["hub", "upload"],
            "experimental",
        )
        assert kind.unavailable_reason is None and kind.materialize is True
        for other in ("audio_classification", "tabular_classification", "video_classification"):
            assert PytorchBackend.custom_model_kinds(get_task_descriptor(other)) == []

    def test_a_worker_may_train_one_because_the_trained_model_is_self_contained(self):
        assert PytorchBackend.remote_custom_models is True

    @pytest.mark.parametrize(
        "config, expected",
        [
            ({"model_type": "vit", "architectures": ["ViTForImageClassification"]}, "hf_image"),
            ({"model_type": "vit", "architectures": ["ViTModel"]}, "hf_image"),
            ({"model_type": "convnext", "architectures": ["ConvNextForImageClassification"]}, "hf_image"),
            ({"model_type": "resnet", "architectures": ["ResNetForImageClassification"]}, "hf_image"),
            ({"model_type": "deit", "architectures": ["DeiTForImageClassificationWithTeacher"]}, None),
            ({"model_type": "clip", "architectures": ["CLIPModel"]}, None),
            ({"model_type": "videomae", "architectures": ["VideoMAEForVideoClassification"]}, None),
            ({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}, None),
            ({"model_type": "vit"}, None),
        ],
    )
    def test_the_architecture_says_which_checkpoints_are_image_models(self, config, expected):
        assert PytorchBackend.suggest_custom_kind(config) == expected

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_supported_family_is_listed_and_its_model_validates(self, family, tmp_path):
        make_tiny(tmp_path, family)
        assert json.loads((tmp_path / "config.json").read_text())["model_type"] in custom.SUPPORTED_MODEL_TYPES
        PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))

    def test_a_bare_backbone_validates_too(self, tmp_path):
        make_tiny(tmp_path, "vit", bare=True)
        PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))

    def test_every_listed_family_has_a_test_row(self):
        # The rows are keyed by `model_type`, which the test above checks against what each tiny model writes.
        assert set(custom.SUPPORTED_MODEL_TYPES) == set(FAMILIES)

    def test_a_kind_that_is_not_ours_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            PytorchBackend.validate_custom_model(TASK, ref_for(tiny, kind="hf_vision"))

    def test_a_task_the_kind_does_not_serve_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            PytorchBackend.validate_custom_model(get_task_descriptor("text_classification"), ref_for(tiny))

    def test_code_that_runs_from_the_repo_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, auto_map={"AutoModel": "x.Y"})
        with pytest.raises(ConfigError, match="custom code"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_without_safetensors_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "model.safetensors").unlink()
        (folder / "pytorch_model.bin").write_bytes(b"pickle")
        with pytest.raises(ConfigError, match="safetensors"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_of_another_family_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, model_type="bert")
        with pytest.raises(ConfigError, match="supported image model"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_weights_that_do_not_match_the_config_are_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, hidden_size=64, intermediate_size=128)
        with pytest.raises(ConfigError, match="cannot be loaded as an image classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_for_greyscale_images_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, num_channels=1)
        with pytest.raises(ConfigError, match="3 channels"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_folder_that_is_not_a_model_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="config.json"):
            PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))


class TestImageSettings:
    def test_the_size_comes_from_the_config_and_the_normalisation_from_the_preprocessor(self, tiny):
        assert custom.image_settings(tiny) == {"imageSize": 32, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    def test_without_a_preprocessor_file_it_is_imagenet(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").unlink()
        found = custom.image_settings(folder)
        assert found["mean"] == [0.485, 0.456, 0.406] and found["std"] == [0.229, 0.224, 0.225]

    def test_an_unreadable_preprocessor_file_falls_back_to_imagenet(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").write_text("{not json")
        assert custom.image_settings(folder)["mean"] == [0.485, 0.456, 0.406]

    def test_a_model_with_no_size_of_its_own_takes_the_preprocessors(self, tmp_path):
        make_tiny(tmp_path, "resnet")
        assert "image_size" not in json.loads((tmp_path / "config.json").read_text())
        for preprocessor, expected in (
            ({"size": {"shortest_edge": 256}}, 256),
            ({"size": {"height": 192, "width": 192}}, 192),
            ({"size": 160}, 160),
            # The crop is what reaches the network, so it wins over the size it was cut from.
            ({"size": {"shortest_edge": 256}, "crop_size": {"height": 224, "width": 224}}, 224),
            ({}, 224),
        ):
            (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor))
            assert custom.image_settings(tmp_path)["imageSize"] == expected

    def test_a_transformers_own_size_wins_over_the_preprocessor(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").write_text(json.dumps({"size": {"height": 224, "width": 224}}))
        assert custom.image_settings(folder)["imageSize"] == 32

    def test_a_preprocessor_that_does_not_normalise_means_no_normalisation(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").write_text(
            json.dumps({"do_normalize": False, "image_mean": [0.5] * 3, "image_std": [0.5] * 3})
        )
        found = custom.image_settings(folder)
        assert found["mean"] == [0.0, 0.0, 0.0] and found["std"] == [1.0, 1.0, 1.0]

    def test_a_model_that_wants_unscaled_pixels_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").write_text(json.dumps({"do_rescale": False}))
        with pytest.raises(ConfigError, match="not scaled"):
            custom.image_settings(folder)

    def test_a_standard_deviation_of_zero_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "preprocessor_config.json").write_text(
            json.dumps({"image_mean": [0.5] * 3, "image_std": [0.0, 0.2, 0.2]})
        )
        with pytest.raises(ConfigError, match="not positive"):
            custom.image_settings(folder)


class TestCompile:
    def test_a_custom_model_is_recorded_for_training_and_ignores_the_image_size(self, tiny):
        config = compile_pytorch_config(
            TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1", image_size=128), custom=ref_for(tiny)
        )
        assert config["model"] == "custom:u1" and config["pretrained"] is True
        assert config["custom"] == {"kind": "hf_image", "path": str(tiny)}
        assert config["classNames"] == CLASSES and config["inputKind"] == "image"
        # Only a stand-in: train.py reads the real size and normalisation from the model's files.
        assert config["image"]["imageSize"] != 128

    def test_the_other_training_knobs_still_apply(self, tiny):
        config = compile_pytorch_config(
            TASK,
            SNAPSHOT,
            PytorchHyperparameters(model_id="custom:u1", epochs=7, freeze_backbone=True, horizontal_flip=False),
            custom=ref_for(tiny),
        )
        assert (config["train"]["epochs"], config["train"]["freezeBackbone"], config["train"]["horizontalFlip"]) == (
            7,
            True,
            False,
        )

    def test_training_from_scratch_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="its own weights"):
            compile_pytorch_config(
                TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1", pretrained=False), custom=ref_for(tiny)
            )

    def test_an_unresolved_custom_id_is_an_error(self):
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:gone"))

    def test_a_custom_model_of_the_wrong_kind_is_an_error(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(
                TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1"), custom=ref_for(tiny, kind="hf_vision")
            )

    def test_a_custom_model_for_another_task_is_an_error(self, tiny):
        text = SnapshotContext(
            columns=[
                ColumnSpec(name="text", kind="inline_text"),
                ColumnSpec(name="class", kind="label"),
                ColumnSpec(name="split", kind="split"),
            ],
            label_class_names=CLASSES,
        )
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(
                get_task_descriptor("text_classification"),
                text,
                PytorchHyperparameters(model_id="custom:u1"),
                custom=ref_for(tiny),
            )

    def test_a_built_in_model_has_no_custom_block(self):
        config = compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters())
        assert "custom" not in config and config["model"] == "resnet18"


def backbone_matches(net, original) -> int:
    """How many tensors of `original` (a saved checkpoint, as a classifier or a bare backbone) are in `net`'s backbone
    with exactly the same values. Fails if any that has a counterpart differs."""
    ours = net.model.state_dict()
    prefix = net.model.base_model_prefix
    matched = 0
    for key, value in original.state_dict().items():
        if key.startswith("classifier."):
            continue
        found = ours.get(key, ours.get(f"{prefix}.{key}"))
        if found is None:
            continue  # e.g. the pooler a classifier's backbone leaves out
        assert np.array_equal(found.cpu().numpy(), value.cpu().numpy()), key
        matched += 1
    return matched


@pytest.mark.parametrize("bare", [False, True], ids=["classifier", "bare backbone"])
@pytest.mark.parametrize("family", FAMILIES)
class TestNetwork:
    def test_the_pretrained_weights_are_really_loaded_and_the_head_is_new(self, family, bare, tmp_path):
        import torch

        original = make_tiny(tmp_path, family, bare=bare)
        net = hf.build(str(tmp_path), CLASSES)
        # Without this a checkpoint that loaded as random weights would pass every other check.
        assert backbone_matches(net, original) >= 10
        out = net(torch.zeros(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 3)
        assert net.model.config.id2label == dict(enumerate(CLASSES))

    def test_freezing_trains_only_the_head(self, family, bare, tmp_path):
        make_tiny(tmp_path, family, bare=bare)
        net = hf.build(str(tmp_path), CLASSES)
        hf.freeze_backbone(net)
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        assert trainable and all(n.startswith("model.classifier.") for n in trainable)
        assert any(not p.requires_grad for p in net.parameters())

    def test_it_exports_to_one_onnx_file_that_agrees(self, family, bare, tmp_path):
        import onnxruntime as ort
        import torch

        make_tiny(tmp_path / "src", family, bare=bare)
        net = hf.build(str(tmp_path / "src"), CLASSES)
        meta = {
            "model": "custom:u1", "hf": True, "imageSize": 32, "mean": [0.5] * 3, "std": [0.25] * 3,
            "imageColumn": "image_path", "outputName": "class", "names": CLASSES, "task": "classify",
            "inputKind": "image",
        }  # fmt: skip
        loaded = wrap(net, meta)
        path = PytorchBackend.convert(loaded, "onnx", str(tmp_path / "export"))
        assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
        batch = torch.randn(3, 3, 32, 32)
        with torch.no_grad():
            expected = loaded.net(batch.to(runtime.device())).cpu().numpy()
        (logits,) = ort.InferenceSession(path).run(["logits"], {"images": batch.numpy()})
        np.testing.assert_allclose(logits, expected, rtol=1e-2, atol=1e-2)

    def test_what_is_saved_loads_back_by_itself(self, family, bare, tmp_path):
        import torch

        make_tiny(tmp_path / "src", family, bare=bare)
        net = hf.build(str(tmp_path / "src"), CLASSES).eval()
        hf.save(net, str(tmp_path / "saved"))
        shutil.rmtree(tmp_path / "src")
        back = hf.load(str(tmp_path / "saved")).eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            np.testing.assert_allclose(back(x).numpy(), net(x).numpy(), rtol=1e-5, atol=1e-5)


class TestStrictLoading:
    def drop_tensor(self, folder: Path, contains: str) -> str:
        from safetensors.torch import load_file, save_file

        tensors = load_file(folder / "model.safetensors")
        victim = sorted(k for k in tensors if contains in k)[0]
        del tensors[victim]
        save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})
        return victim

    def test_a_classifier_missing_a_backbone_tensor_is_refused_not_re_initialised(self, tmp_path):
        make_tiny(tmp_path, "vit")
        self.drop_tensor(tmp_path, "layernorm_after")
        with pytest.raises(ValueError, match="no weights for"):
            hf.build(str(tmp_path), CLASSES)

    def test_a_bare_backbone_missing_a_tensor_is_refused(self, tmp_path):
        make_tiny(tmp_path, "vit", bare=True)
        self.drop_tensor(tmp_path, "layernorm_after")
        with pytest.raises(ValueError, match="no weights for"):
            hf.build(str(tmp_path), CLASSES)

    def test_a_trained_model_missing_its_head_is_refused(self, tmp_path):
        make_tiny(tmp_path / "src", "vit")
        hf.save(hf.build(str(tmp_path / "src"), CLASSES), str(tmp_path / "saved"))
        self.drop_tensor(tmp_path / "saved", "classifier")
        with pytest.raises(ValueError, match="incomplete"):
            hf.load(str(tmp_path / "saved"))

    def test_validation_reports_a_missing_tensor_as_a_model_problem(self, tmp_path):
        make_tiny(tmp_path, "vit")
        self.drop_tensor(tmp_path, "layernorm_after")
        with pytest.raises(ConfigError, match="cannot be loaded as an image classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))


def write_image(path: Path, label: str, rng: np.random.Generator) -> None:
    """A flat colour per class with noise, deliberately not square so the shorter-edge crop is exercised."""
    pixels = np.zeros((48, 72, 3), dtype=np.float32)
    pixels[:, :] = COLOURS[label]
    pixels += rng.normal(0, 25, pixels.shape)
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Fine-tunes the tiny ViT on three colours, then deletes the files it started from."""
    tmp_path = tmp_path_factory.mktemp("custom_run")
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        for label in CLASSES:
            for i in range(count):
                item = f"{split}-{label}-{i}"
                write_image(source / f"{item}.png", label, rng)
                rows.append(
                    {"image_path": f"s3://theseus-datasets/pool/{item}.png", "class": label, "split": split,
                     "_theseus_item_id": item}
                )  # fmt: skip
    df = pd.DataFrame(rows)

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(source / os.path.basename(key), local_path)
        return local_path

    uploaded = tmp_path / "uploaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(data.storage, "download_file", fake_download)
    mp.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    mp.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    mp.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    # A private copy of the model's files, deleted afterwards to prove the trained model does not need them.
    original = tmp_path / "original"
    shutil.copytree(tiny, original)
    config = compile_pytorch_config(
        TASK, SNAPSHOT,
        PytorchHyperparameters(model_id="custom:u1", epochs=6, batch_size=8, learning_rate=0.003, seed=1),
        custom=ref_for(original),
    )  # fmt: skip
    reports: list = []
    ctx = TrainContext(
        run_id="hf", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )  # fmt: skip
    model = PytorchBackend.train(ctx)
    shutil.rmtree(original)
    yield types.SimpleNamespace(
        model=model, reports=reports, uploaded=uploaded, source=source, df=df, tmp_path=tmp_path, config=config
    )
    mp.undo()


def test_training_saves_a_self_contained_hugging_face_folder(trained):
    assert isinstance(trained.model, ImageModel)
    assert {"config.json", "model.safetensors", "names.json"} <= {p.name for p in trained.uploaded.iterdir()}
    assert not (trained.uploaded / "model.pt").exists()
    assert not (trained.tmp_path / "work").exists()
    meta = json.loads((trained.uploaded / "names.json").read_text())
    # The model dictated the image, whatever the config asked for, and the new 3-class head replaced the 5-class one.
    assert (meta["hf"], meta["imageSize"], meta["mean"], meta["std"], meta["names"]) == (
        True, 32, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], CLASSES
    )  # fmt: skip
    assert meta["model"] == "custom:u1" and meta["inputKind"] == "image"
    assert trained.model.output.labels == CLASSES and trained.model.spec.size == 32


def test_training_reports_each_epoch_and_learns(trained):
    epochs_run = [e for e, s, _ in trained.reports if s == "validation"]
    assert epochs_run == list(range(1, len(epochs_run) + 1)) and 1 <= len(epochs_run) <= 6
    assert all("loss" in m and "accuracy" in m for _, _, m in trained.reports)
    assert max(m["accuracy"] for _, s, m in trained.reports if s == "validation") > 0.6


def test_a_trained_model_loads_with_the_original_files_gone(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    assert isinstance(loaded, PytorchLoadedModel) and loaded.output.labels == CLASSES
    assert loaded.spec.size == 32 and loaded.spec.mean == (0.5, 0.5, 0.5)


def test_predict_and_evaluate_from_the_uploaded_files(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    frame = loaded.predict(pd.DataFrame({"image_path": [str(trained.source / "test-green-0.png")]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    result = PytorchBackend.evaluate(loaded, trained.df, "split", "_theseus_item_id")
    assert result is not None and result.report["rowCount"] == 12 and result.report["idx2str"] == CLASSES


def test_the_export_agrees_with_the_platform_through_the_generated_client(trained, tmp_path):
    """A bring-your-own model ships exactly like a built-in one: the same manifest, the same single ONNX file, and the
    shipped client's own preprocessing and softmax give the platform's probabilities."""
    import onnxruntime as ort

    loaded = PytorchBackend.load(str(trained.uploaded))
    path = get_backend("pytorch").convert(loaded, "onnx", str(tmp_path / "export"))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    session = ort.InferenceSession(path)
    assert session.get_inputs()[0].name == "images" and session.get_inputs()[0].shape[1:] == [3, 32, 32]

    module = types.ModuleType("omnitrain_client_custom")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copyfile(path, bundle / "model.onnx")
    meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
    manifest = build_preprocessing_manifest("hf", {}, meta)
    assert manifest["inputs"][0]["imageResize"]["mode"] == "shorter_edge_center_crop"
    (bundle / "preprocessing.json").write_text(json.dumps(manifest))
    client = module.OmniTrainClient(bundle / "model.onnx", bundle / "preprocessing.json")
    for name in ("test-blue-1.png", "test-red-2.png"):
        image = trained.source / name
        from_client = client.predict(str(image), top_k=3)
        from_platform = loaded.golden_prediction(loaded.predict(pd.DataFrame({"image_path": [str(image)]})))
        assert from_client.keys() == from_platform.keys()
        for label, p in from_client.items():
            assert p == pytest.approx(from_platform[label], abs=2e-3)


def test_freezing_the_backbone_trains_a_custom_model_and_leaves_it_untouched(tiny, tmp_path, monkeypatch):
    """With the backbone frozen, only the new head's weights differ from the checkpoint after a run."""
    import torch

    rng = np.random.default_rng(1)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for label in CLASSES:
            for i in range(count):
                item = f"{split}-{label}-{i}"
                write_image(source / f"{item}.png", label, rng)
                rows.append({"image_path": f"s3://b/{item}.png", "class": label, "split": split,
                             "_theseus_item_id": item})  # fmt: skip
    df = pd.DataFrame(rows)

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(source / os.path.basename(key), local_path)
        return local_path

    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(data.storage, "download_file", fake_download)
    monkeypatch.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")
    config = compile_pytorch_config(
        TASK, SNAPSHOT,
        PytorchHyperparameters(model_id="custom:u1", epochs=2, batch_size=4, freeze_backbone=True, seed=2),
        custom=ref_for(tiny),
    )  # fmt: skip
    ctx = TrainContext(
        run_id="hf-frozen", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None, _report=lambda *a: None,
    )  # fmt: skip
    PytorchBackend.train(ctx)

    from safetensors.torch import load_file

    before = load_file(tiny / "model.safetensors")
    after = load_file(uploaded / "model.safetensors")
    body = [k for k in after if not k.startswith("classifier.")]
    assert body and all(torch.equal(before[k], after[k]) for k in body)
