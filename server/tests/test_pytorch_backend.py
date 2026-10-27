"""The PyTorch trainer backend without training anything: compile, image preprocessing, the manifest, registration."""

import types
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pydantic import ValidationError

import omnitrain.plugins.export
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef
from omnitrain.plugins.backends.pytorch import PytorchBackend, audio, data, images, tabular, text
from omnitrain.plugins.backends.pytorch.compile import (
    PytorchHyperparameters,
    compile_pytorch_config,
    hyperparameter_specs,
)
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, ImageSpec, TabularSpec, TextSpec
from omnitrain.plugins.backends.registry import get_backend, trainable_backends
from omnitrain.plugins.export.formats._kits import supports_generated_clients
from omnitrain.plugins.export.registry import list_export_formats

TASK = get_task_descriptor("image_classification")
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="image_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["cat", "dog", "bird"],
)


def compile_(ctx: SnapshotContext = SNAPSHOT, **selection):
    return compile_pytorch_config(TASK, ctx, PytorchHyperparameters(**selection))


class TestRegistration:
    def test_the_backend_trains_images_tables_sounds_texts_and_tags_only(self):
        assert get_backend("pytorch") is PytorchBackend
        supported_tasks = (
            "image_classification",
            "tabular_classification",
            "tabular_regression",
            "audio_classification",
            "text_classification",
            "token_classification",
        )
        for supported in supported_tasks:
            assert PytorchBackend.supports(get_task_descriptor(supported))
        assert not PytorchBackend.supports(get_task_descriptor("video_classification"))
        assert not PytorchBackend.supports(get_task_descriptor("object_detection"))
        assert not PytorchBackend.supports(get_task_descriptor("text_generation"))
        assert PytorchBackend in trainable_backends(TASK)

    def test_it_offers_the_image_classifiers_by_these_ids(self):
        ids = [m.id for m in PytorchBackend.models(TASK)]
        assert ids == ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_small", "convnext_tiny", "vit_base"]
        assert all(m.pretrained for m in PytorchBackend.models(TASK))
        assert PytorchBackend.models(get_task_descriptor("video_classification")) == []

    def test_listing_it_never_imports_torch(self):
        import subprocess
        import sys

        code = (
            "import sys; from omnitrain.plugins.backends.registry import list_backends; "
            "[b.available() for b in list_backends()]; print('torch' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_it_is_the_default_for_images_sounds_and_texts_but_not_tables(self):
        assert PytorchBackend.default_for == {"image_classification", "audio_classification", "text_classification"}
        for task_id in PytorchBackend.default_for:
            task = get_task_descriptor(task_id)
            assert trainable_backends(task)[0] is PytorchBackend
        for task_id in ("tabular_classification", "tabular_regression"):
            assert trainable_backends(get_task_descriptor(task_id))[0] is get_backend("sklearn")

    def test_it_offers_three_perceptrons_for_a_table(self):
        for task_id in ("tabular_classification", "tabular_regression"):
            models = PytorchBackend.models(get_task_descriptor(task_id))
            assert [m.id for m in models] == ["mlp_medium", "mlp_small", "mlp_large"]
            assert not any(m.pretrained for m in models)

    def test_every_export_format_that_takes_a_classifier_is_offered_for_it(self):
        assert supports_generated_clients(TASK)
        offered = {f.id for f in list_export_formats(TASK, PytorchBackend)}
        assert {"python_devkit", "typescript_devkit"} <= offered

    def test_hyperparameter_specs_cover_every_knob_the_model_accepts(self):
        names = {p.name for p in hyperparameter_specs(TASK)}
        assert {"imageSize", "pretrained", "freezeBackbone", "horizontalFlip", "optimizer"} <= names
        assert {"seed", "weightDecay", "lrSchedule", "warmupEpochs", "gradClipNorm", "mixedPrecision"} <= names
        assert hyperparameter_specs(get_task_descriptor("video_classification")) == []


class TestCompile:
    def test_defaults_come_from_the_first_model(self):
        c = compile_()
        assert (c["backend"], c["task"], c["model"], c["pretrained"]) == ("pytorch", "classify", "resnet18", True)
        assert (c["imageColumn"], c["labelColumn"], c["classNames"]) == ("image_path", "class", ["cat", "dog", "bird"])
        assert c["image"]["imageSize"] == 224 and c["image"]["mean"][0] == pytest.approx(0.485)

    def test_the_spec_in_the_config_round_trips(self):
        spec = ImageSpec.from_meta(compile_(image_size=160)["image"])
        assert spec.size == 160 and spec.mean == (0.485, 0.456, 0.406) and spec.std == (0.229, 0.224, 0.225)

    def test_explicit_selections_override_defaults(self):
        c = compile_(model_id="resnet50", image_size=128, epochs=3, batch_size=8, optimizer="SGD", pretrained=False)
        assert c["model"] == "resnet50" and c["image"]["imageSize"] == 128 and c["pretrained"] is False
        assert (c["train"]["epochs"], c["train"]["batchSize"], c["train"]["optimizer"]) == (3, 8, "SGD")

    def test_auto_batch_size_falls_back_to_a_default(self):
        assert compile_(batch_size="auto")["train"]["batchSize"] == 32

    def test_a_disabled_early_stop_can_never_trigger(self):
        assert compile_(epochs=12, early_stop_patience=-1)["train"]["patience"] == 12

    def test_seed_is_unset_unless_given(self):
        assert compile_()["train"]["seed"] is None
        assert compile_(seed=7)["train"]["seed"] == 7

    def test_optimisation_selections_reach_the_config(self):
        train = compile_(
            weight_decay=0.05, lr_schedule="cosine", warmup_epochs=1.5, grad_clip_norm=1.0, mixed_precision=True
        )["train"]
        assert (train["weightDecay"], train["lrSchedule"], train["warmupEpochs"]) == (0.05, "cosine", 1.5)
        assert (train["gradClipNorm"], train["mixedPrecision"]) == (1.0, True)
        defaults = compile_()["train"]
        assert defaults["weightDecay"] is None and defaults["lrSchedule"] == "constant"

    def test_a_vision_transformer_only_works_at_its_own_size(self):
        assert compile_(model_id="vit_base")["image"]["imageSize"] == 224
        with pytest.raises(ConfigError, match="only works at 224"):
            compile_(model_id="vit_base", image_size=160)

    def test_unknown_model_is_a_config_error(self):
        with pytest.raises(ConfigError, match="not an image model"):
            compile_(model_id="mc3_18")

    def test_needs_at_least_two_classes(self):
        one = SnapshotContext(columns=SNAPSHOT.columns, label_class_names=["cat"])
        with pytest.raises(ConfigError, match="two classes"):
            compile_(one)

    def test_snapshot_without_a_label_column_is_a_config_error(self):
        bare = SnapshotContext(columns=[SNAPSHOT.columns[0]], label_class_names=["a", "b"])
        with pytest.raises(ConfigError, match='"label" column'):
            compile_(bare)

    def test_another_task_is_refused(self):
        with pytest.raises(ConfigError, match="does not support the task"):
            compile_pytorch_config(get_task_descriptor("video_classification"), SNAPSHOT, PytorchHyperparameters())

    def test_a_custom_model_of_another_kind_or_an_unresolved_one_is_refused(self):
        # What a good `hf_image` model compiles to is in tests/test_pytorch_custom.py.
        other = CustomModelRef(id="u", kind="hf_transformer", source_kind="hub", local_path="unused")
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(), custom=other)
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_(model_id="custom:abc")

    def test_unknown_hyperparameters_are_rejected(self):
        with pytest.raises(ValidationError):
            PytorchHyperparameters(imgsz=224)  # type: ignore[call-arg]


CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"
SPEC = ImageSpec(size=32, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@pytest.fixture(scope="module")
def client_module():
    # Compiled from source rather than imported, so no __pycache__ is written next to the template.
    module = types.ModuleType("omnitrain_client_under_test")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    return module


def write_image(path, width, height, seed=0):
    pixels = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


# landscape, portrait, square, odd leftovers (round-half-to-even), tiny sides that are upscaled, extreme ratios
SHAPES = [(100, 60), (60, 100), (64, 64), (301, 224), (225, 448), (33, 500), (500, 33), (50, 30), (35, 32), (32, 35)]


class TestImagesMatchTheGeneratedClients:
    """What the platform does to an image and what a generated client does must be the same, bit for bit: that is the
    point of one `ImageSpec`. The reference is the shipped Python client itself."""

    @pytest.mark.parametrize(("width", "height"), SHAPES)
    def test_the_geometry_is_the_clients(self, client_module, width, height):
        assert images.shorter_edge_geometry(width, height, 32) == client_module._shorter_edge_geometry(
            width, height, 32
        )

    @pytest.mark.parametrize(("width", "height"), SHAPES)
    def test_the_tensor_is_the_clients(self, client_module, tmp_path, width, height):
        path = write_image(tmp_path / "a.png", width, height, seed=width + height)
        client = object.__new__(client_module.OmniTrainClient)
        manifest = build_preprocessing_manifest("r", {}, _meta())
        client.input_spec = manifest["inputs"][0]
        expected = client._preprocess_image(path)[0]
        np.testing.assert_array_equal(images.preprocess(str(path), SPEC), expected)

    def test_a_non_rgb_image_is_converted(self, tmp_path):
        Image.fromarray(np.full((20, 30), 128, dtype=np.uint8)).save(tmp_path / "grey.png")
        out = images.preprocess(str(tmp_path / "grey.png"), SPEC)
        assert out.shape == (3, 32, 32) and out.dtype == np.float32
        assert np.allclose(out[0], (128 / 255 - 0.485) / 0.229, atol=1e-4)


class TestAugmentation:
    def test_it_gives_the_network_input_shape_and_is_repeatable_for_one_seed(self, tmp_path):
        path = str(write_image(tmp_path / "a.png", 90, 60))
        first = images.augment(path, SPEC, np.random.default_rng([1, 2, 3]), flip=True)
        again = images.augment(path, SPEC, np.random.default_rng([1, 2, 3]), flip=True)
        other = images.augment(path, SPEC, np.random.default_rng([1, 3, 3]), flip=True)
        assert first.shape == (3, 32, 32) and first.dtype == np.float32
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_flip_off_never_mirrors(self, tmp_path):
        # A left-half-white image: with the crop pinned to the whole picture a mirror would move the white to the right.
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, :16] = 255
        Image.fromarray(pixels).save(tmp_path / "half.png")
        for seed in range(8):
            out = images.augment(str(tmp_path / "half.png"), SPEC, np.random.default_rng(seed), flip=False)
            assert out[0, :, 0].mean() >= out[0, :, -1].mean()

    def test_a_thin_image_falls_back_to_the_whole_picture(self, tmp_path):
        path = str(write_image(tmp_path / "thin.png", 400, 4))
        assert images.augment(path, SPEC, np.random.default_rng(0), flip=True).shape == (3, 32, 32)


class TestMaterialize:
    def test_unreadable_unlabelled_and_unknown_rows_are_skipped_not_fatal(self, tmp_path, monkeypatch):
        good = write_image(tmp_path / "good.png", 20, 20)
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")

        def fake_download(bucket, key, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes((bad if "bad" in key else good).read_bytes())
            return local_path

        monkeypatch.setattr(data.storage, "download_file", fake_download)
        rows = [
            ("s3://b/good1.png", "cat", "train", "i1"),
            ("s3://b/good2.png", "dog", "validation", "i2"),
            ("s3://b/bad.png", "cat", "train", "i3"),  # downloads but cannot be decoded
            ("s3://b/good3.png", None, "train", "i4"),  # unlabelled
            ("s3://b/good4.png", "emu", "train", "i5"),  # a class the model does not know
            ("s3://b/good5.png", "cat", "weird", "i6"),  # no such split
        ]
        df = pd.DataFrame(rows, columns=["image_path", "class", "split", "_theseus_item_id"])
        result = data.materialize(
            df, str(tmp_path / "dest"), file_column="image_path", label_column="class", split_column="split",
            item_id_column="_theseus_item_id", class_names=["cat", "dog", "bird"],
        )  # fmt: skip
        assert [s.class_index for s in result.samples["train"]] == [0]
        assert [s.class_index for s in result.samples["val"]] == [1]
        assert result.samples["test"] == [] and result.skipped == 4

    def test_a_split_that_ends_up_empty_is_an_error(self, tmp_path, monkeypatch):

        def fake_download(bucket, key, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            return write_image(Path(local_path), 8, 8)

        monkeypatch.setattr(data.storage, "download_file", fake_download)
        df = pd.DataFrame(
            [("s3://b/a.png", "cat", "train", "i1")], columns=["image_path", "class", "split", "_theseus_item_id"]
        )
        with pytest.raises(ValueError, match="validation split has no usable images"):
            data.materialize(
                df, str(tmp_path / "d"), file_column="image_path", label_column="class", split_column="split",
                item_id_column="_theseus_item_id", class_names=["cat", "dog"],
            )  # fmt: skip


def _meta() -> dict:
    return {
        **SPEC.to_meta(),
        "task": "classify",
        "imageColumn": "image_path",
        "outputName": "class",
        "names": ["cat", "dog", "bird"],
    }


class TestManifest:
    def test_it_describes_what_the_clients_must_do(self):
        m = build_preprocessing_manifest("run-1", {}, _meta())
        [image] = m["inputs"]
        assert image["type"] == "image" and image["column"] == "image_path"
        assert image["imageResize"] == {"mode": "shorter_edge_center_crop", "interpolation": "bilinear"}
        assert image["imageNormalization"] == {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        assert image["imageSize"] == {"height": 32, "width": 32}

    def test_the_class_list_is_the_models_own_order(self):
        [out] = build_preprocessing_manifest("run-1", {}, _meta())["outputs"]
        assert out["classes"] == ["cat", "dog", "bird"] and out["type"] == "category"

    def test_without_the_metadata_it_refuses_rather_than_guessing(self):
        with pytest.raises(ValueError, match="names.json"):
            build_preprocessing_manifest("run-1", {}, None)


TABULAR = get_task_descriptor("tabular_classification")
REGRESSION = get_task_descriptor("tabular_regression")
TABLE = SnapshotContext(
    columns=[
        ColumnSpec(name="age", kind="scalar"),
        ColumnSpec(name="income", kind="scalar"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["no", "yes"],
)
REGRESSION_TABLE = SnapshotContext(
    columns=[
        ColumnSpec(name="rooms", kind="scalar"),
        ColumnSpec(name="area", kind="scalar"),
        ColumnSpec(name="target", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=[],
)


def compile_table(task=TABULAR, ctx: SnapshotContext = TABLE, **selection):
    return compile_pytorch_config(task, ctx, PytorchHyperparameters(**selection))


class TestCompileTable:
    def test_every_scalar_column_is_a_feature(self):
        c = compile_table()
        assert (c["backend"], c["inputKind"], c["task"], c["model"]) == ("pytorch", "tabular", "classify", "mlp_medium")
        assert (c["inputColumns"], c["labelColumn"], c["classNames"]) == (["age", "income"], "class", ["no", "yes"])
        assert c["network"] == {"hidden": [128, 128, 64], "dropout": 0.1}

    def test_a_regression_has_no_classes(self):
        c = compile_table(REGRESSION, REGRESSION_TABLE)
        assert (c["task"], c["classNames"], c["labelColumn"]) == ("regress", [], "target")
        assert c["train"]["useClassWeights"] is False

    def test_the_model_and_dropout_choose_the_network(self):
        c = compile_table(model_id="mlp_large", dropout=0.3)
        assert c["network"] == {"hidden": [256, 256, 128, 64], "dropout": 0.3}
        assert compile_table(dropout=0)["network"]["dropout"] == 0

    def test_defaults_suit_a_table(self):
        train = compile_table()["train"]
        assert (train["epochs"], train["batchSize"], train["learningRate"], train["patience"]) == (60, 128, 0.001, 10)
        assert compile_table(batch_size="auto", early_stop_patience=-1, epochs=7)["train"]["patience"] == 7

    def test_class_weights_are_a_classification_knob(self):
        assert compile_table(use_class_weights=True)["train"]["useClassWeights"] is True
        with pytest.raises(ConfigError, match="useClassWeights"):
            compile_table(REGRESSION, REGRESSION_TABLE, use_class_weights=True)

    def test_a_knob_of_the_other_kind_of_input_is_refused_not_ignored(self):
        for knob in ({"image_size": 128}, {"pretrained": False}, {"freeze_backbone": True}, {"horizontal_flip": False}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_table(**knob)
        for knob in ({"dropout": 0.2}, {"use_class_weights": True}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_(**knob)

    def test_unknown_model_no_columns_and_one_class_are_config_errors(self):
        with pytest.raises(ConfigError, match="not a table model"):
            compile_table(model_id="resnet18")
        no_columns = SnapshotContext(columns=[TABLE.columns[2], TABLE.columns[3]], label_class_names=["a", "b"])
        with pytest.raises(ConfigError, match="numeric columns"):
            compile_table(ctx=no_columns)
        with pytest.raises(ConfigError, match="two classes"):
            compile_table(ctx=SnapshotContext(columns=TABLE.columns, label_class_names=["a"]))

    def test_the_forms_differ_by_kind(self):
        table = {p.name for p in hyperparameter_specs(TABULAR)}
        assert {"dropout", "useClassWeights", "epochs", "seed", "weightDecay", "lrSchedule"} <= table
        assert not {"imageSize", "pretrained", "freezeBackbone", "horizontalFlip"} & table
        regression = {p.name for p in hyperparameter_specs(REGRESSION)}
        assert "dropout" in regression and "useClassWeights" not in regression
        assert not {"dropout", "useClassWeights"} & {p.name for p in hyperparameter_specs(TASK)}


class TestTabularData:
    def test_anything_that_is_not_a_finite_number_is_nan(self):
        frame = pd.DataFrame({"a": [1, "x", None, float("inf"), "2.5"], "b": [1.0, 2.0, 3.0, 4.0, -float("inf")]})
        x = tabular.features(frame, ["a", "b"])
        assert x.dtype == np.float32 and x.shape == (5, 2)
        assert np.isnan(x).tolist() == [[False, False], [True, False], [True, False], [True, False], [False, True]]
        assert x[4, 0] == pytest.approx(2.5)

    def test_the_statistics_come_from_the_train_rows_alone(self):
        df = pd.DataFrame(
            {
                "age": [10.0, 20.0, 30.0, 1000.0, -500.0, 7000.0],
                "income": [1.0, 3.0, np.nan, 5.0, 5.0, 5.0],
                "class": ["no", "yes", "no", "yes", "yes", "no"],
                "split": ["train", "train", "train", "validation", "validation", "test"],
            }
        )
        splits = tabular.prepare(
            df, kind="classify", columns=["age", "income"], label_column="class", class_names=["no", "yes"],
            split_column="split",
        )  # fmt: skip
        assert [len(splits[s]) for s in ("train", "validation", "test")] == [3, 2, 1]
        spec = tabular.fit_spec(["age", "income"], splits["train"].x, None)
        assert spec.mean == pytest.approx((20.0, 2.0)) and spec.fill == pytest.approx((20.0, 2.0))
        assert spec.std[0] == pytest.approx(np.std([10.0, 20.0, 30.0]))
        # The validation and test rows (1000, -500, 7000) moved none of it.
        assert spec.std[1] == pytest.approx(1.0)

    def test_labels_are_class_indices_in_the_given_order_and_unknown_ones_are_left_out(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "class": ["yes", "no", "emu", None], "split": ["train"] * 4})
        split = tabular.prepare(
            df, kind="classify", columns=["a"], label_column="class", class_names=["no", "yes"], split_column="split"
        )["train"]
        assert split.y.tolist() == [1, 0] and split.x[:, 0].tolist() == [1.0, 2.0]

    def test_a_regression_target_must_be_a_finite_number(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [5.0, "oops", float("inf")], "split": ["train"] * 3})
        split = tabular.prepare(
            df, kind="regress", columns=["a"], label_column="target", class_names=[], split_column="split"
        )["train"]
        assert split.y.tolist() == [5.0] and split.y.dtype == np.float32

    def test_constant_empty_and_single_row_columns_are_safe(self):
        x = np.array([[3.0, np.nan, 1.0], [3.0, np.nan, 2.0]], dtype=np.float32)
        spec = tabular.fit_spec(["c", "e", "v"], x, np.array([4.0, 4.0], dtype=np.float32))
        assert spec.std[0] == 1.0 and spec.mean[1] == 0.0 and spec.fill[1] == 0.0 and spec.std[1] == 1.0
        assert spec.target_mean == 4.0 and spec.target_std == 1.0
        one = tabular.fit_spec(["a"], np.array([[5.0]], dtype=np.float32), None)
        assert one.mean == (5.0,) and one.std == (1.0,)

    def test_balanced_weights_favour_the_rare_class_and_skip_an_absent_one(self):
        weights = tabular.balanced_class_weights(np.array([0, 0, 0, 1], dtype=np.int64), 3)
        assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1), 0.0])


class TestTabularManifest:
    def meta(self, **extra) -> dict:
        return {
            "task": "classify", "inputKind": "tabular", "inputColumns": ["age", "income"], "outputName": "class",
            "names": ["no", "yes"], **extra,
        }  # fmt: skip

    def test_a_classifier_takes_raw_numbers_and_has_no_client_side_scaling(self):
        m = build_preprocessing_manifest("r", {}, self.meta())
        assert m["inputs"] == [
            {"name": "age", "type": "number", "column": "age"},
            {"name": "income", "type": "number", "column": "income"},
        ]
        assert not any("numberNormalization" in i for i in m["inputs"])
        assert m["outputs"] == [{"name": "class", "type": "category", "column": "class", "classes": ["no", "yes"]}]

    def test_a_regressor_outputs_a_number(self):
        m = build_preprocessing_manifest("r", {}, self.meta(task="regress", names=[], outputName="target"))
        assert m["outputs"] == [{"name": "target", "type": "number", "column": "target"}]


class TestTabularNetwork:
    def test_it_fills_standardises_and_un_standardises_inside_the_module(self):
        torch = pytest.importorskip("torch")
        from omnitrain.plugins.backends.pytorch import network

        spec = TabularSpec(
            columns=("a", "b"), mean=(10.0, 0.0), std=(2.0, 1.0), fill=(10.0, 5.0), target_mean=100.0, target_std=10.0
        )
        net = network.build_tabular(spec, [4], 0.0, 1, regress=True).eval()
        # Zero every weight and bias, then a final bias of 1.0: the network's (standardised) output is exactly 1.0 for
        # any input, so the result shows only the un-standardising: 1.0 * 10 + 100.
        for p in net.layers.parameters():
            torch.nn.init.zeros_(p)
        net.layers[-1].bias.data.fill_(1.0)
        out = net(torch.tensor([[12.0, 1.0], [float("nan"), float("inf")]]))
        assert out.shape == (2,) and out.tolist() == pytest.approx([110.0, 110.0])

        # And the input side: a cell that is not finite becomes the column's fill before standardising.
        nan_row = torch.tensor([[float("nan"), float("nan")]])
        filled = torch.where(torch.isfinite(nan_row), nan_row, net.fill)
        assert ((filled - net.mean) / net.std).tolist() == [[0.0, 5.0]]

    def test_a_classifier_outputs_one_logit_per_class(self):
        torch = pytest.importorskip("torch")
        from omnitrain.plugins.backends.pytorch import network

        spec = TabularSpec(columns=("a",), mean=(0.0,), std=(1.0,), fill=(0.0,))
        net = network.build_tabular(spec, [8, 8], 0.1, 3, regress=False).eval()
        assert net(torch.zeros(5, 1)).shape == (5, 3)


AUDIO = get_task_descriptor("audio_classification")
SOUNDS = SnapshotContext(
    columns=[
        ColumnSpec(name="audio_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["dog", "siren"],
)


def compile_audio(ctx: SnapshotContext = SOUNDS, **selection):
    return compile_pytorch_config(AUDIO, ctx, PytorchHyperparameters(**selection))


def write_wav(path: Path, samples: np.ndarray, rate: int) -> str:
    """`samples` is `(frames,)` or `(frames, channels)` in [-1, 1], written as 16-bit PCM."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
    return str(path)


class TestCompileAudio:
    def test_defaults(self):
        c = compile_audio()
        assert (c["backend"], c["inputKind"], c["task"], c["model"], c["pretrained"]) == (
            "pytorch", "audio", "classify", "audio_resnet18", True,
        )  # fmt: skip
        assert (c["audioColumn"], c["labelColumn"], c["classNames"]) == ("audio_path", "class", ["dog", "siren"])
        spec = AudioSpec.from_meta(c["audio"])
        assert (spec.sample_rate, spec.seconds, spec.n_mels, spec.clips) == (16000, 5.0, 64, 3)
        assert spec.frames == 1 + (80000 - 1024) // 256
        assert c["train"]["specAugment"] is True and c["train"]["freezeBackbone"] is False
        assert (c["train"]["epochs"], c["train"]["batchSize"], c["train"]["learningRate"]) == (30, 16, 0.0005)

    def test_the_audio_knobs_reach_the_spec(self):
        c = compile_audio(sample_rate=8000, clip_seconds=2.5, n_mels=40, clips_per_audio=1, spec_augment=False)
        spec = AudioSpec.from_meta(c["audio"])
        assert (spec.sample_rate, spec.seconds, spec.n_mels, spec.clips) == (8000, 2.5, 40, 1)
        assert c["train"]["specAugment"] is False

    def test_the_scratch_model_cannot_be_pretrained(self):
        assert compile_audio(model_id="audio_cnn")["pretrained"] is False
        with pytest.raises(ConfigError, match="no pretrained weights"):
            compile_audio(model_id="audio_cnn", pretrained=True)
        assert compile_audio(pretrained=False)["pretrained"] is False

    def test_a_window_too_short_for_the_network_is_refused(self):
        with pytest.raises(ConfigError, match="at least 32"):
            compile_audio(sample_rate=4000, clip_seconds=1)

    def test_unknown_model_and_one_class_are_config_errors(self):
        with pytest.raises(ConfigError, match="not an audio model"):
            compile_audio(model_id="resnet18")
        with pytest.raises(ConfigError, match="two classes"):
            compile_audio(SnapshotContext(columns=SOUNDS.columns, label_class_names=["dog"]))

    def test_a_knob_of_another_kind_of_input_is_refused_both_ways(self):
        for knob in ({"image_size": 128}, {"horizontal_flip": False}, {"dropout": 0.2}, {"use_class_weights": True}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_audio(**knob)
        for knob in ({"sample_rate": 8000}, {"clip_seconds": 3}, {"n_mels": 40}, {"spec_augment": False}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_(**knob)
            with pytest.raises(ConfigError, match="does not support"):
                compile_table(**knob)

    def test_the_form_lists_the_sound_knobs_and_the_models(self):
        names = {p.name for p in hyperparameter_specs(AUDIO)}
        assert {"sampleRate", "clipSeconds", "nMels", "clipsPerAudio", "specAugment", "pretrained", "epochs"} <= names
        assert not {"imageSize", "horizontalFlip", "dropout", "useClassWeights"} & names
        models = PytorchBackend.models(AUDIO)
        assert [(m.id, m.pretrained) for m in models] == [("audio_resnet18", True), ("audio_cnn", False)]


class TestAudioRecipe:
    SPEC = AudioSpec(sample_rate=16000, seconds=1.0, n_mels=40)

    def test_a_wav_is_decoded_mixed_down_and_resampled(self, tmp_path):
        t = np.arange(8000) / 8000
        left, right = np.sin(2 * np.pi * 440 * t), np.zeros_like(t)
        path = write_wav(tmp_path / "a.wav", np.stack([left, right], axis=1), 8000)
        samples, rate = audio.decode(path)
        assert (samples.shape, rate) == ((2, 8000), 8000)
        mono = audio.load_waveform(path, self.SPEC)
        assert mono.dtype == np.float32 and len(mono) == 16000
        # Averaging the channels halves the loudness of a tone that is only on one of them.
        assert np.abs(mono).max() == pytest.approx(0.5, abs=0.02)

    def test_an_unreadable_file_raises(self, tmp_path):
        bad = tmp_path / "x.wav"
        bad.write_bytes(b"not audio at all")
        with pytest.raises(Exception):  # noqa: B017 - scipy and torchaudio each raise their own
            audio.load_waveform(str(bad), self.SPEC)

    def test_windows_cover_a_long_clip_and_pad_a_short_one(self):
        assert audio.window_starts(500, 1000, 3) == [0]
        assert audio.window_starts(1000, 1000, 3) == [0]
        assert audio.window_starts(5000, 1000, 1) == [2000]
        assert audio.window_starts(5000, 1000, 3) == [0, 2000, 4000]
        assert audio.window_starts(1001, 1000, 3) == [0, 1]
        short = audio.cut(np.ones(300, dtype=np.float32), 0, 1000)
        assert len(short) == 1000 and short[:300].sum() == 300 and short[300:].sum() == 0
        assert len(audio.cut(np.ones(5000, dtype=np.float32), 4500, 1000)) == 1000

    def test_a_random_window_stays_inside_the_clip(self):
        rng = np.random.default_rng(0)
        starts = {audio.random_start(5000, 1000, rng) for _ in range(200)}
        assert min(starts) >= 0 and max(starts) <= 4000 and len(starts) > 10
        assert audio.random_start(500, 1000, rng) == 0

    def test_the_spectrogram_has_the_size_the_network_is_built_for(self):
        window = np.random.default_rng(0).standard_normal(self.SPEC.length).astype(np.float32) * 0.1
        out = audio.spectrogram(window, self.SPEC)
        assert out.shape == (40, self.SPEC.frames) and out.dtype == np.float32
        assert np.isfinite(audio.spectrogram(np.zeros(self.SPEC.length, dtype=np.float32), self.SPEC)).all()

    def test_it_matches_torchaudio(self):
        torch = pytest.importorskip("torch")
        torchaudio = pytest.importorskip("torchaudio")
        spec = AudioSpec(sample_rate=16000, seconds=1.0, n_mels=64)
        window = np.random.default_rng(1).standard_normal(spec.length).astype(np.float32) * 0.2
        ours = audio.log_mel(window, spec)
        filters = torchaudio.functional.melscale_fbanks(
            spec.n_fft // 2 + 1, spec.f_min, spec.f_max, spec.n_mels, spec.sample_rate, norm=None, mel_scale="htk"
        )
        assert np.allclose(audio.mel_filterbank(16000, spec.n_fft, 64, 0.0, 8000.0), filters.numpy(), atol=1e-4)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=spec.sample_rate, n_fft=spec.n_fft, hop_length=spec.hop, f_min=spec.f_min, f_max=spec.f_max,
            n_mels=spec.n_mels, window_fn=torch.hann_window, power=2.0, center=False, norm=None, mel_scale="htk",
        )  # fmt: skip
        theirs = torch.log(mel(torch.from_numpy(window)) + audio.LOG_FLOOR).numpy()
        assert theirs.shape == ours.shape
        assert np.allclose(ours, theirs, rtol=1e-3, atol=1e-3)

    def test_spec_augment_masks_and_does_not_touch_the_original(self):
        base = np.ones((64, 100), dtype=np.float32)
        out = audio.spec_augment(base, np.random.default_rng(3))
        assert base.min() == 1.0 and out.min() == 0.0 and out.shape == base.shape
        assert np.array_equal(out, audio.spec_augment(base, np.random.default_rng(3)))


class TestAudioManifest:
    def test_the_recipe_is_spelled_out_for_a_client(self):
        spec = AudioSpec(sample_rate=16000, seconds=5.0, n_mels=64, mean=-4.5, std=2.0)
        meta = {
            "task": "classify", "model": "audio_cnn", **spec.to_meta(), "audioColumn": "audio_path",
            "outputName": "class", "names": ["dog", "siren"],
        }  # fmt: skip
        m = build_preprocessing_manifest("r", {}, meta)
        [i] = m["inputs"]
        assert (i["name"], i["type"], i["column"]) == ("spectrograms", "audio", "audio_path")
        recipe = i["audioSpectrogram"]
        assert (recipe["sampleRate"], recipe["windowSamples"], recipe["frames"]) == (16000, 80000, spec.frames)
        assert (recipe["stft"]["nFft"], recipe["stft"]["hopLength"], recipe["stft"]["center"]) == (1024, 256, False)
        assert recipe["mel"]["bands"] == 64 and recipe["mel"]["fMax"] == 8000.0
        assert recipe["normalization"] == {"mean": -4.5, "std": 2.0}
        assert m["outputs"] == [{"name": "class", "type": "category", "column": "class", "classes": ["dog", "siren"]}]


TEXT = get_task_descriptor("text_classification")
REVIEWS = SnapshotContext(
    columns=[
        ColumnSpec(name="text", kind="inline_text"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["negative", "positive"],
)


def compile_text(ctx: SnapshotContext = REVIEWS, **selection):
    return compile_pytorch_config(TEXT, ctx, PytorchHyperparameters(**selection))


class TestTokenizer:
    def test_it_lower_cases_ascii_letters_only_and_splits_on_ascii_punctuation(self):
        assert text.tokenize("Hello, World! It's 3.14_ok") == ["hello", "world", "it", "s", "3", "14", "ok"]

    def test_a_non_ascii_letter_is_part_of_a_token_and_is_never_case_folded(self):
        # Vietnamese stays whole; the ASCII letters of "Nẵng" fold but "Đ" (not ASCII) does not.
        assert text.tokenize("Xin chào, Việt Nam! Đà Nẵng") == ["xin", "chào", "việt", "nam", "Đà", "nẵng"]
        assert text.tokenize("ÀÉ") == ["ÀÉ"]

    def test_unicode_spaces_separate_words_too(self):
        assert text.tokenize("a\u00a0b\u2003c\u3000d\u202fe") == ["a", "b", "c", "d", "e"]

    def test_nothing_to_read_gives_no_tokens(self):
        assert text.tokenize("") == [] and text.tokenize(" \t\n,.;") == []

    def test_the_separator_ranges_are_exactly_what_splits(self):
        # The manifest hands these ranges to every client, so they must agree with the regex the platform uses.
        for code in list(range(0, 0x300)) + [0x2000, 0x200A, 0x200B, 0x2028, 0x202F, 0x3000, 0x3001, 0xFEFF]:
            char = chr(code)
            if "A" <= char <= "Z":
                continue
            separates = any(lo <= code <= hi for lo, hi in text.SEPARATOR_RANGES)
            assert (text.tokenize(f"x{char}x") == ["x", "x"]) == separates, hex(code)


class TestVocabulary:
    def test_it_keeps_the_frequent_words_most_frequent_first_and_alphabetical_among_equals(self):
        texts = ["b a a c", "c b a", "d"]
        assert text.fit_vocabulary(texts, max_words=10, min_count=1) == ["a", "b", "c", "d"]
        assert text.fit_vocabulary(texts, max_words=10, min_count=2) == ["a", "b", "c"]
        assert text.fit_vocabulary(texts, max_words=2, min_count=1) == ["a", "b"]

    def test_the_same_texts_always_give_the_same_vocabulary(self):
        texts = [f"w{i % 7} w{i % 5} w{i % 3}" for i in range(50)]
        assert text.fit_vocabulary(texts, max_words=5, min_count=1) == text.fit_vocabulary(
            list(reversed(texts)), max_words=5, min_count=1
        )

    def test_a_word_is_counted_every_time_it_occurs(self):
        assert text.fit_vocabulary(["spam spam spam eggs"], max_words=5, min_count=2) == ["spam"]


class TestEncode:
    SPEC = TextSpec(max_tokens=5, vocab=("good", "bad"))

    def test_ids_are_the_vocabulary_position_plus_two_and_the_end_is_padded(self):
        assert text.encode(["Good bad GOOD"], self.SPEC).tolist() == [[2, 3, 2, 0, 0]]

    def test_an_unknown_word_is_one_and_a_long_text_is_cut(self):
        assert text.encode(["good meh bad good good good bad"], self.SPEC).tolist() == [[2, 1, 3, 2, 2]]

    def test_an_empty_text_is_all_padding_and_the_shape_is_fixed(self):
        out = text.encode(["", "  ", "bad"], self.SPEC)
        assert out.shape == (3, 5) and out.dtype == np.int64
        assert out.tolist() == [[0] * 5, [0] * 5, [3, 0, 0, 0, 0]]

    def test_the_embedding_table_has_room_for_padding_and_unknown(self):
        assert self.SPEC.size == 4


class TestTextSpec:
    def test_it_round_trips_through_names_json(self):
        spec = TextSpec(max_tokens=64, vocab=("a", "b"))
        assert TextSpec.from_meta(spec.to_meta()) == spec
        assert spec.to_meta()["inputKind"] == "text"

    def test_a_vocabulary_can_be_attached_afterwards(self):
        assert TextSpec(max_tokens=8).with_vocabulary(["x"]).vocab == ("x",)


class TestTextData:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": ["great film", "awful film", "meh", None, "  ", "great", "awful", "great", "awful"],
                "class": ["positive", "negative", "positive", "positive", "positive", None, "other", "positive", None],
                "split": ["train", "train", "train", "train", "train", "train", "train", "validation", "validation"],
            }
        )

    def test_a_row_needs_a_known_class_and_a_text_with_something_in_it(self):
        parts = text.prepare(
            self.frame(), text_column="text", label_column="class", class_names=["negative", "positive"],
            split_column="split",
        )  # fmt: skip
        assert parts["train"].texts == ["great film", "awful film", "meh"]
        assert parts["train"].y.tolist() == [1, 0, 1]
        assert parts["validation"].texts == ["great"] and parts["validation"].y.tolist() == [1]
        assert len(parts["test"]) == 0 and parts["test"].y.dtype == np.int64

    def test_a_missing_text_scores_as_the_empty_string(self):
        assert text.clean(pd.Series(["a", None, float("nan"), 3])) == ["a", "", "", "3"]


class TestCompileText:
    def test_defaults(self):
        c = compile_text()
        assert (c["backend"], c["inputKind"], c["task"], c["model"]) == ("pytorch", "text", "classify", "text_bag")
        assert (c["textColumn"], c["labelColumn"], c["classNames"]) == ("text", "class", ["negative", "positive"])
        assert c["text"] == {"maxTokens": 128, "vocabSize": 20000, "minWordCount": 2}
        assert c["network"] == {"dropout": 0.3}
        assert c["train"]["useClassWeights"] is False and c["train"]["optimizer"] == "AdamW"

    def test_the_selection_is_honoured(self):
        c = compile_text(
            model_id="text_cnn", max_tokens=64, vocab_size=500, min_word_count=1, dropout=0.5, use_class_weights=True,
            epochs=3, batch_size=16, learning_rate=0.01,
        )  # fmt: skip
        assert c["model"] == "text_cnn" and c["text"] == {"maxTokens": 64, "vocabSize": 500, "minWordCount": 1}
        assert c["network"]["dropout"] == 0.5 and c["train"]["useClassWeights"] is True
        assert (c["train"]["epochs"], c["train"]["batchSize"], c["train"]["learningRate"]) == (3, 16, 0.01)

    def test_a_text_model_has_no_pretrained_weights_to_start_from(self):
        # The form always offers the freeze switch (a bring-your-own model uses it), so only turning it ON is refused.
        for selection in ({"pretrained": True}, {"freeze_backbone": True}):
            with pytest.raises(ConfigError, match="trained from scratch"):
                compile_text(**selection)
        assert compile_text(freeze_backbone=False)["model"] == "text_bag"

    def test_knobs_of_the_other_kinds_of_input_are_refused(self):
        for selection in ({"image_size": 64}, {"horizontal_flip": True}, {"sample_rate": 8000}, {"n_mels": 40}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_text(**selection)

    def test_the_text_knobs_are_refused_for_every_other_kind(self):
        for selection in ({"max_tokens": 64}, {"vocab_size": 500}, {"min_word_count": 3}):
            with pytest.raises(ConfigError, match="does not support"):
                compile_(**selection)
            with pytest.raises(ConfigError, match="does not support"):
                compile_audio(**selection)
            with pytest.raises(ConfigError, match="does not support"):
                compile_pytorch_config(
                    get_task_descriptor("tabular_classification"), TABLE, PytorchHyperparameters(**selection)
                )

    def test_an_unknown_or_custom_model_is_refused(self):
        with pytest.raises(ConfigError, match="not a text model"):
            compile_text(model_id="resnet18")
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_text(model_id="custom:00000000-0000-0000-0000-000000000000")
        # The only text kind is `hf_text` (tests/test_pytorch_custom_text.py); any other is for another task.
        for kind in ("hf_transformer", "hf_image"):
            with pytest.raises(ConfigError, match="cannot be used"):
                compile_pytorch_config(
                    TEXT, REVIEWS, PytorchHyperparameters(),
                    CustomModelRef(id="u", kind=kind, source_kind="hub", local_path="unused"),
                )  # fmt: skip

    def test_a_snapshot_with_one_class_or_no_text_column_is_refused(self):
        with pytest.raises(ConfigError, match="at least two classes"):
            compile_text(SnapshotContext(columns=REVIEWS.columns, label_class_names=["only"]))
        with pytest.raises(ConfigError, match="inline_text"):
            compile_text(
                SnapshotContext(columns=[REVIEWS.columns[1], REVIEWS.columns[2]], label_class_names=["a", "b"])
            )

    def test_the_form_offers_the_text_knobs_and_none_of_the_others(self):
        names = {p.name for p in hyperparameter_specs(TEXT)}
        assert {"maxTokens", "vocabSize", "minWordCount", "dropout", "useClassWeights", "optimizer"} <= names
        assert {"seed", "weightDecay", "lrSchedule", "warmupEpochs", "gradClipNorm", "mixedPrecision"} <= names
        # `freezeBackbone` is there for a bring-your-own transformer; the built-in models refuse to turn it on.
        assert "freezeBackbone" in names
        assert not names & {"imageSize", "pretrained", "sampleRate", "nMels", "specAugment"}

    def test_it_offers_a_bag_and_a_cnn_and_neither_is_pretrained(self):
        models = PytorchBackend.models(TEXT)
        assert [m.id for m in models] == ["text_bag", "text_cnn"]
        assert not any(m.pretrained for m in models)

    def test_it_is_the_default_for_text_and_sklearn_stays_offered(self):
        offered = trainable_backends(TEXT)
        assert PytorchBackend in offered and get_backend("sklearn") in offered
        assert offered[0] is PytorchBackend


class TestTextManifest:
    META = {
        "task": "classify", "model": "text_bag", **TextSpec(max_tokens=16, vocab=("good", "bad", "ok")).to_meta(),
        "textColumn": "review", "network": {"dropout": 0.3}, "outputName": "class", "names": ["neg", "pos"],
    }  # fmt: skip

    def test_the_recipe_is_spelled_out_for_a_client(self):
        m = build_preprocessing_manifest("r", {}, self.META)
        [i] = m["inputs"]
        assert (i["name"], i["type"], i["column"], i["layout"]) == (
            "tokens",
            "text",
            "review",
            "[batch, maxTokens] int64",
        )
        recipe = i["textTokens"]
        assert recipe["recipe"] == "word_ids_v1"
        assert (recipe["maxTokens"], recipe["padId"], recipe["unknownId"], recipe["firstWordId"]) == (16, 0, 1, 2)
        assert recipe["vocabulary"] == ["good", "bad", "ok"]
        assert recipe["lowercase"] == {"from": 65, "to": 90, "add": 32}
        assert recipe["separatorRanges"] == [list(r) for r in text.SEPARATOR_RANGES]
        assert m["outputs"] == [{"name": "class", "type": "category", "column": "class", "classes": ["neg", "pos"]}]

    def test_it_is_not_scikit_learns_string_input(self):
        [i] = build_preprocessing_manifest("r", {}, self.META)["inputs"]
        assert "textInput" not in i

    def test_the_class_order_comes_from_the_trained_model_not_the_table(self):
        meta = {**self.META, "names": ["pos", "neg"]}
        assert build_preprocessing_manifest("r", {}, meta)["outputs"][0]["classes"] == ["pos", "neg"]
