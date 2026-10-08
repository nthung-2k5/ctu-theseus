import pytest
from pydantic import ValidationError
from pydantic.alias_generators import to_camel

from theseus.backends.base import ConfigError
from theseus.backends.yolo import YoloBackend
from theseus.backends.yolo.compile import YoloHyperparameters, compile_yolo_config, hyperparameter_specs
from theseus.services.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

VISION = SnapshotContext(
    columns=[
        ColumnSpec(name="image_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=["cat", "dog"],
)
TASK = get_task_descriptor("image_classification")


def compile_(ctx: SnapshotContext = VISION, **sel):
    return compile_yolo_config(TASK, ctx, YoloHyperparameters(**sel))


class TestCompile:
    def test_defaults_pick_the_first_pretrained_model(self):
        config = compile_()
        assert (config["task"], config["model"], config["weights"]) == ("classify", "yolo11n-cls", "yolo11n-cls.pt")
        assert config["train"]["pretrained"] is True
        assert config["train"]["imgsz"] == 224
        assert config["imageColumn"] == "image_path"
        assert config["labelColumn"] == "class"
        assert config["classNames"] == ["cat", "dog"]

    def test_learning_rate_always_comes_with_an_explicit_optimizer(self):
        # `optimizer="auto"` would make Ultralytics ignore lr0, so a sweep over it would do nothing.
        train = compile_()["train"]
        assert train["optimizer"] == "AdamW"
        assert train["lr0"] == 0.001
        train = compile_(optimizer="SGD", learning_rate=0.02)["train"]
        assert (train["optimizer"], train["lr0"]) == ("SGD", 0.02)

    def test_explicit_selections_override_defaults(self):
        config = compile_(model_id="yolo11s-cls", epochs=5, batch_size=32, image_size=320, freeze_layers=4, seed=7)
        assert config["model"] == "yolo11s-cls"
        assert config["train"] == {
            "epochs": 5,
            "batch": 32,
            "imgsz": 320,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "patience": 10,
            "pretrained": True,
            "freeze": 4,
            "seed": 7,
        }

    def test_auto_batch_size_is_ultralytics_autobatch(self):
        assert compile_(batch_size="auto")["train"]["batch"] == -1

    def test_a_disabled_early_stop_can_never_trigger(self):
        assert compile_(epochs=20, early_stop_patience=-1)["train"]["patience"] == 20

    def test_not_pretrained_uses_the_architecture_file(self):
        config = compile_(pretrained=False)
        assert config["weights"] == "yolo11n-cls.yaml"
        assert config["train"]["pretrained"] is False

    def test_unknown_model_is_a_config_error(self):
        with pytest.raises(ConfigError, match="yolo99"):
            compile_(model_id="yolo99")

    def test_image_size_must_be_a_multiple_of_32(self):
        with pytest.raises(ConfigError, match="multiple of 32"):
            compile_(image_size=250)

    def test_needs_at_least_two_classes(self):
        one_class = SnapshotContext(columns=VISION.columns, label_class_names=["cat"])
        with pytest.raises(ConfigError, match="two classes"):
            compile_(one_class)

    def test_snapshot_without_a_label_column_is_a_config_error(self):
        no_label = SnapshotContext(columns=VISION.columns[:1], label_class_names=["a", "b"])
        with pytest.raises(ConfigError, match="label"):
            compile_(no_label)

    def test_unsupported_task_and_custom_models_are_refused(self):
        with pytest.raises(ConfigError, match="does not support"):
            compile_yolo_config(get_task_descriptor("tabular_classification"), VISION, YoloHyperparameters())
        with pytest.raises(ConfigError, match="bring-your-own"):
            compile_yolo_config(TASK, VISION, YoloHyperparameters(), custom=object())  # type: ignore[arg-type]


class TestHyperparameters:
    def test_wire_names_are_camel_case_and_unknown_keys_are_rejected(self):
        hp = YoloHyperparameters.model_validate({"modelId": "yolo11m-cls", "imageSize": 320, "freezeLayers": 2})
        assert (hp.model_id, hp.image_size, hp.freeze_layers) == ("yolo11m-cls", 320, 2)
        # Ludwig-only knobs must not be accepted: the web form would otherwise send them to this backend.
        with pytest.raises(ValidationError):
            YoloHyperparameters.model_validate({"freezeBackbone": False})

    def test_every_form_spec_is_a_real_hyperparameter_and_its_default_validates(self):
        specs = hyperparameter_specs(TASK)
        known = {to_camel(name) for name in YoloHyperparameters.model_fields}
        assert {s.name for s in specs} <= known
        # The form sends choice values back as strings; they must validate as-is.
        defaults = {s.name: s.default for s in specs}
        assert defaults["batchSize"] == "16"
        assert defaults["imageSize"] == "224"
        YoloHyperparameters.model_validate(defaults)

    def test_specs_are_empty_for_a_task_the_backend_does_not_train(self):
        assert hyperparameter_specs(get_task_descriptor("tabular_classification")) == []


class TestBackend:
    def test_supports_classification_and_detection_only(self):
        assert YoloBackend.supports(TASK)
        assert YoloBackend.supports(get_task_descriptor("object_detection"))
        assert not YoloBackend.supports(get_task_descriptor("tabular_classification"))
        assert not YoloBackend.supports(get_task_descriptor("image_segmentation"))

    def test_models_are_the_pretrained_catalog(self):
        models = YoloBackend.models(TASK)
        assert [m.id for m in models][0] == "yolo11n-cls"
        assert all(m.pretrained for m in models)
        assert YoloBackend.models(get_task_descriptor("tabular_classification")) == []

    def test_listing_the_backend_does_not_import_ultralytics(self):
        import subprocess
        import sys

        code = (
            "import sys; from theseus.backends.yolo import YoloBackend; "
            "YoloBackend.available(); assert 'ultralytics' not in sys.modules, 'imported ultralytics'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
