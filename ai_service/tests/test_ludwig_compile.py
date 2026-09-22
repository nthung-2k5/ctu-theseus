import math

import pytest
import yaml
from pydantic import ValidationError

from theseus.backends.base import ConfigError
from theseus.backends.ludwig.compile import LudwigHyperparameters, compile_ludwig_config, serialize_ludwig_config
from theseus.backends.ludwig.tasks import LUDWIG_TASKS
from theseus.services.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor


def cols(*pairs: tuple[str, str]) -> list[ColumnSpec]:
    return [ColumnSpec(name=n, kind=k) for n, k in pairs]


VISION = SnapshotContext(
    columns=cols(("image_path", "storage_uri"), ("class", "label"), ("split", "split")),
    label_class_names=["cat", "dog"],
)
CAPTIONING = SnapshotContext(
    columns=cols(("image_path", "storage_uri"), ("caption", "text_sequence_label"), ("split", "split"))
)
ASR = SnapshotContext(
    columns=cols(("audio_path", "storage_uri"), ("transcript", "text_sequence_label"), ("split", "split"))
)
TABULAR = SnapshotContext(
    columns=cols(("age", "scalar"), ("income", "scalar"), ("class", "label"), ("split", "split")),
    label_class_names=["approved", "denied"],
)


def compile_(task: str, ctx: SnapshotContext, **sel):
    return compile_ludwig_config(get_task_descriptor(task), ctx, LudwigHyperparameters(**sel))


class TestCompile:
    def test_vision_task_gets_declared_input_and_default_encoder(self):
        config = compile_("image_classification", VISION)
        assert config["model_type"] == "ecd"
        assert len(config["input_features"]) == 1
        f = config["input_features"][0]
        assert (f["name"], f["type"], f["column"]) == ("image_path", "image", "image_path")
        assert f["encoder"]["type"] == "resnet"  # first vision encoder is the default
        assert config["output_features"] == [{"name": "class", "type": "category", "column": "class"}]

    def test_split_is_pinned_to_the_synthetic_integer_column(self):
        config = compile_("image_classification", VISION)
        assert config["preprocessing"] == {"split": {"type": "fixed", "column": "_ludwig_split_idx"}}

    def test_tabular_input_features_come_from_scalar_columns(self):
        config = compile_("tabular_classification", TABULAR)
        assert sorted(f["name"] for f in config["input_features"]) == ["age", "income"]
        assert all(f["type"] == "number" for f in config["input_features"])

    def test_trainer_selections_override_task_defaults(self):
        config = compile_("image_classification", VISION, epochs=5, batch_size=16, learning_rate=0.01)
        assert config["trainer"]["epochs"] == 5
        assert config["trainer"]["batch_size"] == 16
        assert config["trainer"]["learning_rate"] == 0.01

    def test_defaults_come_from_knobs(self):
        t = compile_("image_classification", VISION)["trainer"]
        assert (t["epochs"], t["batch_size"], t["learning_rate"], t["early_stop"]) == (20, "auto", 0.001, 5)
        assert compile_("text_generation", SnapshotContext())["trainer"]["epochs"] == 3  # llm knobs

    def test_selects_encoder_by_id(self):
        config = compile_("image_classification", VISION, model_id="resnet50")
        assert config["input_features"][0]["encoder"]["type"] == "resnet"
        assert config["input_features"][0]["encoder"]["model_variant"] == 50

    def test_unknown_encoder_raises(self):
        with pytest.raises(ConfigError, match="Unknown encoder"):
            compile_("image_classification", VISION, model_id="not-a-real-encoder")

    def test_tabular_snapshot_without_scalar_columns_raises(self):
        empty = SnapshotContext(columns=cols(("class", "label"), ("split", "split")), label_class_names=["a", "b"])
        with pytest.raises(ConfigError):
            compile_("tabular_classification", empty)

    def test_unsupported_task_raises(self):
        with pytest.raises(ConfigError, match="no Ludwig backend"):
            compile_("object_detection", SnapshotContext())

    def test_registry_is_not_mutated_by_compilation(self):
        compile_("image_classification", VISION, image_size=64)
        assert "preprocessing" not in LUDWIG_TASKS["image_classification"].input_features[0]


class TestClassWeighting:
    def test_untouched_when_not_requested(self):
        assert compile_("image_classification", VISION)["output_features"][0].get("loss") is None

    def test_name_keyed_balanced_weights(self):
        ctx = SnapshotContext(
            columns=VISION.columns, label_class_names=["cat", "dog"], class_counts={"cat": 80, "dog": 20}
        )
        loss = compile_("image_classification", ctx, use_class_weights=True)["output_features"][0]["loss"]
        w = loss["class_weights"]
        assert sorted(w) == ["cat", "dog"]  # names, never indices
        assert math.isclose(w["dog"], 100 / (2 * 20))
        assert math.isclose(w["cat"], 100 / (2 * 80))
        assert w["dog"] > w["cat"]

    def test_balanced_dataset_gets_all_ones(self):
        ctx = SnapshotContext(columns=VISION.columns, class_counts={"cat": 50, "dog": 50})
        w = compile_("image_classification", ctx, use_class_weights=True)["output_features"][0]["loss"]["class_weights"]
        assert math.isclose(w["cat"], 1) and math.isclose(w["dog"], 1)

    def test_empty_class_counts_raises_even_for_regression(self):
        ctx = SnapshotContext(columns=cols(("age", "scalar"), ("target", "label"), ("split", "split")), class_counts={})
        with pytest.raises(ConfigError):
            compile_("tabular_regression", ctx, use_class_weights=True)

    def test_missing_distribution_raises(self):
        with pytest.raises(ConfigError, match="class distribution"):
            compile_("image_classification", VISION, use_class_weights=True)

    def test_task_without_class_distribution_raises_fast(self):
        with pytest.raises(ConfigError, match="class weighting requires"):
            compile_("image_captioning", CAPTIONING, use_class_weights=True)


class TestResize:
    def test_untouched_by_default(self):
        f = compile_("image_classification", VISION)["input_features"][0]
        assert "augmentation" not in f and "preprocessing" not in f

    def test_training_config_never_carries_augmentation(self):
        # Augmentation happens at snapshot creation now, not training: there is no knob for it on
        # LudwigHyperparameters, and (extra="forbid") a stale client that still sends the old field
        # is rejected clearly rather than silently switching Ludwig train-time augmentation back on.
        assert "augmentations" not in LudwigHyperparameters.model_fields
        with pytest.raises(ValidationError):
            LudwigHyperparameters(augmentations=["random_rotate"])

    def test_resize_is_a_noop_for_non_image_tasks(self):
        config = compile_("tabular_classification", TABULAR, image_size=128)
        for f in config["input_features"]:
            assert "augmentation" not in f and "preprocessing" not in f

    def test_resize_sets_height_and_width(self):
        f = compile_("image_classification", VISION, image_size=128)["input_features"][0]
        assert f["preprocessing"]["height"] == 128 and f["preprocessing"]["width"] == 128


class TestValidationMetricAndOptimizer:
    def test_validation_metric_passes_through_or_is_omitted(self):
        assert (
            compile_("image_classification", VISION, validation_metric="accuracy")["trainer"]["validation_metric"]
            == "accuracy"
        )
        assert "validation_metric" not in compile_("image_classification", VISION)["trainer"]

    def test_optimizer_is_set_omitted_or_rejected(self):
        assert compile_("image_classification", VISION, optimizer="adamw")["trainer"]["optimizer"] == {"type": "adamw"}
        assert "optimizer" not in compile_("image_classification", VISION)["trainer"]
        with pytest.raises(ConfigError, match="unknown optimizer"):
            compile_("image_classification", VISION, optimizer="not_a_real_optimizer")


class TestHyperparameterSpecs:
    def test_vision_classification_task_gets_the_full_knob_set(self):
        from theseus.backends.ludwig.compile import hyperparameter_specs

        names = {p.name for p in hyperparameter_specs(get_task_descriptor("image_classification"))}
        assert names == {
            "epochs", "batchSize", "learningRate", "earlyStopPatience", "optimizer",
            "validationMetric", "imageSize", "useClassWeights",
        }  # fmt: skip

    def test_epoch_and_batch_size_defaults_come_from_the_task_not_a_shared_constant(self):
        from theseus.backends.ludwig.compile import hyperparameter_specs

        ecd = {p.name: p for p in hyperparameter_specs(get_task_descriptor("image_classification"))}
        llm = {p.name: p for p in hyperparameter_specs(get_task_descriptor("text_generation"))}
        assert ecd["epochs"].default == 20 and llm["epochs"].default == 3
        assert ecd["batchSize"].default == "auto" and llm["batchSize"].default == "1"

    def test_regression_task_gets_regression_metrics_and_no_class_weights_or_image_size(self):
        from theseus.backends.ludwig.compile import hyperparameter_specs

        specs = {p.name: p for p in hyperparameter_specs(get_task_descriptor("tabular_regression"))}
        assert specs["validationMetric"].choices == ["loss", "mean_squared_error", "mean_absolute_error", "r2"]
        assert "useClassWeights" not in specs and "imageSize" not in specs

    def test_experimental_llm_task_has_no_validation_metric_menu(self):
        from theseus.backends.ludwig.compile import hyperparameter_specs

        specs = {p.name: p for p in hyperparameter_specs(get_task_descriptor("text_generation"))}
        assert "validationMetric" not in specs

    def test_unsupported_task_gets_no_specs(self):
        from theseus.backends.ludwig.compile import hyperparameter_specs

        assert hyperparameter_specs(get_task_descriptor("object_detection")) == []


class TestTier3Tasks:
    def test_image_captioning_pairs_image_input_with_text_output(self):
        config = compile_("image_captioning", CAPTIONING)
        assert config["model_type"] == "ecd"
        assert [(f["name"], f["type"]) for f in config["input_features"]] == [("image_path", "image")]
        assert config["output_features"] == [{"name": "caption", "type": "text", "column": "caption"}]

    def test_audio_captioning_and_asr(self):
        cap = compile_("audio_captioning", CAPTIONING)
        assert [(f["name"], f["type"]) for f in cap["input_features"]] == [("audio_path", "audio")]
        assert cap["output_features"] == [{"name": "caption", "type": "text", "column": "caption"}]
        asr = compile_("automatic_speech_recognition", ASR)
        assert asr["output_features"] == [{"name": "transcript", "type": "text", "column": "transcript"}]


class TestSerialize:
    def test_yaml_roundtrips_and_preserves_key_order(self):
        config = compile_("image_classification", VISION, learning_rate=0.00001, batch_size=32)
        text = serialize_ludwig_config(config)
        assert yaml.safe_load(text) == config
        assert text.index("model_type") < text.index("input_features") < text.index("trainer")
        # Ludwig reads this with a YAML 1.1 loader, so small floats must stay floats, not strings.
        assert isinstance(yaml.safe_load(text)["trainer"]["learning_rate"], float)
