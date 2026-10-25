"""The scikit-learn backend: compile, the form, real training on small synthetic data (scikit-learn is fast enough
that nothing here is faked but S3), serving, evaluation and the model file's safety."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from pydantic.alias_generators import to_camel

from omnitrain.core import constants as C
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor
from omnitrain.plugins.backends.base import ConfigError, TrainContext
from omnitrain.plugins.backends.registry import get_backend, trainable_backends
from omnitrain.plugins.backends.scikit import SklearnBackend, estimators
from omnitrain.plugins.backends.scikit import train as train_mod
from omnitrain.plugins.backends.scikit.compile import (
    SklearnHyperparameters,
    compile_sklearn_config,
    hyperparameter_specs,
)

TABULAR = get_task_descriptor("tabular_classification")
REGRESSION = get_task_descriptor("tabular_regression")
TEXT = get_task_descriptor("text_classification")


def cols(*pairs: tuple[str, str]) -> list[ColumnSpec]:
    return [ColumnSpec(name=n, kind=k) for n, k in pairs]


TABLE = SnapshotContext(
    columns=cols(("x1", "scalar"), ("x2", "scalar"), ("class", "label"), ("split", "split")),
    label_class_names=["low", "mid", "high"],
)
REGRESSION_TABLE = SnapshotContext(
    columns=cols(("x1", "scalar"), ("x2", "scalar"), ("target", "label"), ("split", "split"))
)
TEXT_TABLE = SnapshotContext(
    columns=cols(("text", "inline_text"), ("class", "label"), ("split", "split")), label_class_names=["neg", "pos"]
)


def compile_(task=TABULAR, ctx=TABLE, **sel):
    return compile_sklearn_config(task, ctx, SklearnHyperparameters(**sel))


class TestRegistration:
    def test_it_trains_tabular_and_text_classification_but_not_images(self):
        assert get_backend("sklearn") is SklearnBackend
        for task in (TABULAR, REGRESSION, TEXT):
            assert SklearnBackend.supports(task) and SklearnBackend in trainable_backends(task)
        assert not SklearnBackend.supports(get_task_descriptor("image_classification"))
        assert not SklearnBackend.supports(get_task_descriptor("tabular_clustering"))

    def test_models_differ_per_task(self):
        assert [m.id for m in SklearnBackend.models(TABULAR)] == [
            "hist_gradient_boosting", "random_forest", "logistic_regression",
        ]  # fmt: skip
        assert [m.id for m in SklearnBackend.models(REGRESSION)][-1] == "ridge"
        assert [m.id for m in SklearnBackend.models(TEXT)] == ["logistic_regression", "naive_bayes"]
        assert SklearnBackend.models(get_task_descriptor("image_classification")) == []

    def test_listing_it_never_imports_scikit_learn(self):
        code = (
            "import sys; from omnitrain.plugins.backends.registry import list_backends; "
            "[b.available() for b in list_backends()]; print('sklearn' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_it_produces_an_onnx_artifact_so_the_onnx_export_formats_are_offered(self):
        from omnitrain.plugins.export.registry import list_export_formats

        assert list(SklearnBackend.artifacts) == ["onnx"]
        formats = list_export_formats(TABULAR, SklearnBackend)
        assert formats and all(f.artifact == "onnx" for f in formats)


class TestCompile:
    def test_tabular_features_are_the_scalar_columns(self):
        config = compile_()
        assert (config["task"], config["model"], config["inputColumns"], config["labelColumn"]) == (
            "classify", "hist_gradient_boosting", ["x1", "x2"], "class",
        )  # fmt: skip
        assert config["params"] == {} and "text" not in config and "seed" not in config

    def test_a_regression_task_regresses(self):
        assert compile_(REGRESSION, REGRESSION_TABLE)["task"] == "regress"

    def test_text_gets_its_vectorizer_settings(self):
        config = compile_(TEXT, TEXT_TABLE, ngram_max=3, analyzer="char_wb", min_df=2, sublinear_tf=False)
        assert config["inputColumns"] == ["text"]
        assert config["text"] == {
            "analyzer": "char_wb", "ngramMax": 3, "maxFeatures": 50_000, "minDf": 2, "sublinearTf": False,
        }  # fmt: skip

    def test_a_knob_reaches_only_the_model_it_belongs_to(self):
        boosting = compile_(n_estimators=50, max_depth=3, learning_rate=0.2, regularization_c=5, alpha=2)["params"]
        assert boosting == {"nEstimators": 50, "maxDepth": 3, "learningRate": 0.2}
        forest = compile_(model_id="random_forest", n_estimators=50, learning_rate=0.2)["params"]
        assert forest == {"nEstimators": 50}
        assert compile_(model_id="logistic_regression", regularization_c=5, n_estimators=9)["params"] == {"c": 5}
        assert compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge", alpha=3)["params"] == {"alpha": 3}

    def test_class_weights_are_balanced_except_for_naive_bayes_and_regression(self):
        assert compile_(use_class_weights=True)["params"] == {"classWeight": "balanced"}
        assert "classWeight" not in compile_(TEXT, TEXT_TABLE, model_id="naive_bayes", use_class_weights=True)["params"]
        assert compile_(REGRESSION, REGRESSION_TABLE, use_class_weights=True)["params"] == {}

    def test_seed_is_carried_when_given(self):
        assert compile_(seed=7)["seed"] == 7

    @pytest.mark.parametrize(
        "field", ["epochs", "batch_size", "early_stop_patience", "weight_decay", "lr_schedule", "warmup_epochs",
                  "grad_clip_norm", "mixed_precision"]
    )  # fmt: skip
    def test_what_a_one_shot_fit_cannot_use_is_refused_not_ignored(self, field):
        value = {"lr_schedule": "linear", "mixed_precision": True, "batch_size": 8}.get(field, 1)
        with pytest.raises(ConfigError, match=to_camel(field)):
            compile_(**{field: value})

    def test_unknown_model_task_and_custom_models_are_refused(self):
        with pytest.raises(ConfigError, match="not a scikit-learn model"):
            compile_(model_id="ridge")  # a regression model, asked for a classification task
        with pytest.raises(ConfigError, match="does not support the task"):
            compile_(get_task_descriptor("image_classification"), TABLE)
        with pytest.raises(ConfigError, match="bring-your-own"):
            compile_sklearn_config(TABULAR, TABLE, SklearnHyperparameters(), custom=object())  # type: ignore[arg-type]

    def test_a_snapshot_without_what_it_needs_is_a_config_error(self):
        with pytest.raises(ConfigError, match="at least two classes"):
            compile_(ctx=SnapshotContext(columns=TABLE.columns, label_class_names=["only"]))
        with pytest.raises(ConfigError, match="numeric columns"):
            compile_(
                ctx=SnapshotContext(columns=cols(("class", "label"), ("split", "split")), label_class_names=["a", "b"])
            )
        with pytest.raises(ConfigError, match="inline_text"):
            compile_(TEXT, SnapshotContext(columns=cols(("class", "label")), label_class_names=["a", "b"]))

    def test_unknown_hyperparameters_are_rejected(self):
        with pytest.raises(ValidationError):
            SklearnHyperparameters.model_validate({"freezeBackbone": True})
        assert SklearnHyperparameters.model_validate({"nEstimators": 5, "maxFeatures": "20000"}).max_features == 20_000


class TestForm:
    def test_every_spec_is_a_real_hyperparameter_and_its_default_validates(self):
        known = {to_camel(name) for name in SklearnHyperparameters.model_fields}
        for task in (TABULAR, REGRESSION, TEXT):
            specs = hyperparameter_specs(task)
            assert {s.name for s in specs} <= known
            SklearnHyperparameters.model_validate({s.name: s.default for s in specs})

    def test_each_task_is_offered_only_the_knobs_of_its_models(self):
        tabular = {s.name for s in hyperparameter_specs(TABULAR)}
        assert {"nEstimators", "maxDepth", "learningRate", "regularizationC", "useClassWeights"} <= tabular
        assert not {"alpha", "analyzer", "ngramMax"} & tabular
        regression = {s.name for s in hyperparameter_specs(REGRESSION)}
        assert "alpha" in regression and not {"useClassWeights", "regularizationC"} & regression
        text = {s.name for s in hyperparameter_specs(TEXT)}
        assert {"analyzer", "ngramMax", "maxFeatures", "minDf", "sublinearTf", "alpha"} <= text
        assert not {"nEstimators", "learningRate"} & text

    def test_none_of_the_refused_shared_fields_is_offered(self):
        names = {s.name for s in hyperparameter_specs(TABULAR)}
        assert not names & {"epochs", "batchSize", "earlyStopPatience", "weightDecay", "lrSchedule", "mixedPrecision"}

    def test_sections_follow_the_form_order(self):
        groups = list(dict.fromkeys(s.group for s in hyperparameter_specs(TEXT)))
        assert groups == ["Model details", "Training", "Data"]

    def test_an_unsupported_task_has_no_form(self):
        assert hyperparameter_specs(get_task_descriptor("image_classification")) == []


# -- Training -----------------------------------------------------------------------------------------------


def table_frame(rows_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    """Three well-separated classes in two numbers; every 5th row validation, every 6th test."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(rows_per_class * 3):
        label = ["low", "mid", "high"][i % 3]
        centre = {"low": 0.0, "mid": 5.0, "high": 10.0}[label]
        split = "validation" if i % 5 == 0 else "test" if i % 6 == 0 else "train"
        rows.append(
            {"x1": centre + rng.normal(0, 0.5), "x2": centre + rng.normal(0, 0.5), "class": label, "split": split,
             C.ITEM_ID_COLUMN_NAME: f"item-{i}"}
        )  # fmt: skip
    return pd.DataFrame(rows)


def regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x1": rng.uniform(0, 10, 200), "x2": rng.uniform(0, 10, 200)})
    frame["target"] = 3 * frame["x1"] - 2 * frame["x2"] + rng.normal(0, 0.1, 200)
    frame["split"] = ["validation" if i % 5 == 0 else "test" if i % 6 == 0 else "train" for i in range(200)]
    frame[C.ITEM_ID_COLUMN_NAME] = [f"item-{i}" for i in range(200)]
    return frame


def text_frame() -> pd.DataFrame:
    positive = ["good great fine", "great film good", "fine good plot", "really great fine acting"]
    negative = ["bad awful poor", "awful film bad", "poor bad plot", "really awful poor acting"]
    rows = []
    for i in range(60):
        pool, label = (positive, "pos") if i % 2 == 0 else (negative, "neg")
        split = "validation" if i % 5 == 0 else "test" if i % 6 == 0 else "train"
        rows.append({"text": pool[i % 4], "class": label, "split": split, C.ITEM_ID_COLUMN_NAME: f"item-{i}"})
    return pd.DataFrame(rows)


class Run:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, config: dict):
        self.reports: list[tuple[int, str, dict]] = []
        self.uploaded: dict[str, bytes] = {}
        self.workdir = tmp_path / "work"

        def fake_upload_dir(bucket, prefix, local_dir, **kw):
            self.bucket, self.prefix = bucket, prefix
            for path in Path(local_dir).rglob("*"):
                if path.is_file():
                    self.uploaded[path.relative_to(local_dir).as_posix()] = path.read_bytes()
            # Keep a copy that outlives the workdir, for loading afterwards.
            self.kept = tmp_path / "kept"
            shutil.copytree(local_dir, self.kept)
            return len(self.uploaded)

        monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: frame)
        monkeypatch.setattr(train_mod.storage, "upload_dir", fake_upload_dir)
        self.ctx = TrainContext(
            run_id="run-1",
            config=config,
            dataset_uri="s3://theseus-datasets/snapshots/v/dataset.parquet",
            output_uri="s3://theseus-training/run-1/results/",
            workdir=str(self.workdir),
            _heartbeat=lambda: None,
            _check_abort=lambda: None,
            _report=lambda epoch, split, metrics: self.reports.append((epoch, split, metrics)),
        )


@pytest.fixture
def run_factory(tmp_path, monkeypatch):
    count = iter(range(1000))
    # Each run in a test gets its own directory, since a test may train several models.
    return lambda frame, config: Run(tmp_path / f"run{next(count)}", monkeypatch, frame, config)


class TestTrainTabularClassification:
    @pytest.mark.parametrize("model_id", ["hist_gradient_boosting", "random_forest", "logistic_regression"])
    def test_every_model_learns_separable_classes(self, run_factory, model_id):
        frame = table_frame()
        run = run_factory(frame, compile_(model_id=model_id, seed=0))
        model = train_mod.train(run.ctx)
        predictions = model.predict(frame[["x1", "x2"]])
        accuracy = (predictions["class_predictions"].to_numpy() == frame["class"].to_numpy()).mean()
        assert accuracy > 0.95

    def test_it_uploads_the_model_and_its_class_order_and_cleans_up(self, run_factory):
        run = run_factory(table_frame(), compile_(seed=0))
        train_mod.train(run.ctx)
        assert sorted(run.uploaded) == ["model.skops", "names.json"]
        assert (run.bucket, run.prefix) == (C.BUCKET_TRAINING, "run-1/results/")
        assert not run.workdir.exists()

    def test_the_class_order_is_the_models_own_not_the_snapshots(self, run_factory):
        run = run_factory(table_frame(), compile_(seed=0))
        model = train_mod.train(run.ctx)
        # The snapshot listed low, mid, high; the fitted model sorts its classes, and it is the model that is trusted.
        assert model.output.labels == ["high", "low", "mid"]
        assert json.loads(run.uploaded["names.json"])["names"] == ["high", "low", "mid"]
        probabilities = model.predict(pd.DataFrame({"x1": [10.0], "x2": [10.0]}))["class_probabilities"].iloc[0]
        assert model.output.labels[int(np.argmax(probabilities))] == "high"

    def test_it_reports_one_epoch_with_a_loss_for_train_and_validation(self, run_factory):
        run = run_factory(table_frame(), compile_(seed=0))
        train_mod.train(run.ctx)
        assert [(e, s) for e, s, _ in run.reports] == [(1, "train"), (1, "validation")]
        for _, _, metrics in run.reports:
            assert {"loss", "accuracy"} <= set(metrics) and metrics["accuracy"] > 0.9

    def test_a_saved_model_loads_again_from_disk(self, run_factory):
        run = run_factory(table_frame(), compile_(seed=0))
        trained = train_mod.train(run.ctx)
        loaded = SklearnBackend.load(str(run.kept))
        frame = table_frame()[["x1", "x2"]]
        assert loaded.output.labels == trained.output.labels
        assert (
            loaded.predict(frame)["class_predictions"].tolist() == trained.predict(frame)["class_predictions"].tolist()
        )

    def test_a_missing_value_is_scored_not_an_error(self, run_factory):
        run = run_factory(table_frame(), compile_(model_id="logistic_regression", seed=0))
        model = train_mod.train(run.ctx)
        out = model.predict(pd.DataFrame({"x1": [np.nan, 10.0], "x2": [10.0, None]}))
        assert len(out) == 2

    def test_serving_shapes(self, run_factory):
        run = run_factory(table_frame(), compile_(seed=0))
        model = train_mod.train(run.ctx)
        predictions = model.predict(pd.DataFrame({"x1": [0.0], "x2": [0.0]}))
        output = model.to_output(predictions, top_k=2)
        assert output["kind"] == "classification" and output["feature"] == "class"
        assert [c["label"] for c in output["classes"]][0] == "low" and len(output["classes"]) == 2
        golden = model.golden_prediction(predictions, threshold=0.5)
        assert list(golden) == ["low"]

    def test_class_weights_are_applied(self, run_factory):
        config = compile_(model_id="logistic_regression", use_class_weights=True, seed=0)
        model = train_mod.train(run_factory(table_frame(), config).ctx)
        assert estimators.final_estimator(model.estimator).class_weight == "balanced"

    def test_a_train_split_with_one_class_is_refused(self, run_factory):
        frame = table_frame()
        frame["class"] = "low"
        with pytest.raises(ConfigError, match="only one class"):
            train_mod.train(run_factory(frame, compile_()).ctx)

    def test_an_empty_train_split_is_refused(self, run_factory):
        frame = table_frame()
        frame["split"] = "test"
        with pytest.raises(ConfigError, match="no labelled rows"):
            train_mod.train(run_factory(frame, compile_()).ctx)

    def test_unlabelled_rows_are_left_out(self, run_factory):
        frame = table_frame()
        frame.loc[frame.index[:10], "class"] = None
        train_mod.train(run_factory(frame, compile_(seed=0)).ctx)

    def test_an_unseen_validation_class_does_not_break_scoring(self, run_factory):
        frame = table_frame()
        frame.loc[(frame["split"] == "validation") & (frame.index % 2 == 0), "class"] = "never-seen"
        run = run_factory(frame, compile_(seed=0))
        train_mod.train(run.ctx)
        assert [s for _, s, _ in run.reports] == ["train", "validation"]

    def test_the_same_seed_gives_the_same_model(self, run_factory):
        a = train_mod.train(run_factory(table_frame(), compile_(model_id="random_forest", seed=3)).ctx)
        b = train_mod.train(run_factory(table_frame(), compile_(model_id="random_forest", seed=3)).ctx)
        frame = table_frame()[["x1", "x2"]]
        assert a.predict(frame)["class_probabilities"].tolist() == b.predict(frame)["class_probabilities"].tolist()


class TestTrainRegression:
    def test_a_linear_target_is_learned_and_reported(self, run_factory):
        frame = regression_frame()
        config = compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge", seed=0)
        run = run_factory(frame, config)
        model = train_mod.train(run.ctx)
        assert model.output.kind == "regression" and model.output.labels is None
        assert [(e, s) for e, s, _ in run.reports] == [(1, "train"), (1, "validation")]
        assert run.reports[1][2]["r2"] > 0.99 and {"loss", "mean_absolute_error"} <= set(run.reports[1][2])
        output = model.to_output(model.predict(pd.DataFrame({"x1": [1.0], "x2": [1.0]})))
        assert output["kind"] == "regression" and output["value"] == pytest.approx(1.0, abs=0.2)

    @pytest.mark.parametrize("model_id", ["hist_gradient_boosting", "random_forest"])
    def test_trees_fit_a_nonlinear_target(self, run_factory, model_id):
        frame = regression_frame()
        frame["target"] = (frame["x1"] > 5).astype(float) * 10
        model = train_mod.train(
            run_factory(frame, compile_(REGRESSION, REGRESSION_TABLE, model_id=model_id, seed=0)).ctx
        )
        predicted = model.predict(frame[["x1", "x2"]])["target_predictions"].to_numpy()
        assert np.abs(predicted - frame["target"].to_numpy()).mean() < 1.0

    def test_golden_prediction_is_the_number(self, run_factory):
        frame = regression_frame()
        model = train_mod.train(run_factory(frame, compile_(REGRESSION, REGRESSION_TABLE, seed=0)).ctx)
        golden = model.golden_prediction(model.predict(frame[["x1", "x2"]].head(1)))
        assert list(golden) == ["target"]


class TestTrainText:
    def test_words_separate_the_classes(self, run_factory):
        run = run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, seed=0))
        model = train_mod.train(run.ctx)
        assert model.output.labels == ["neg", "pos"]
        out = model.predict(pd.DataFrame({"text": ["a great and good film", "an awful bad film", None]}))
        assert out["class_predictions"].tolist()[:2] == ["pos", "neg"]
        assert len(out) == 3

    def test_naive_bayes_and_character_ngrams_train_too(self, run_factory):
        config = compile_(TEXT, TEXT_TABLE, model_id="naive_bayes", analyzer="char_wb", ngram_max=3)
        model = train_mod.train(run_factory(text_frame(), config).ctx)
        assert model.predict(pd.DataFrame({"text": ["great"]}))["class_predictions"].iloc[0] == "pos"

    def test_the_longest_phrase_setting_changes_the_vocabulary(self, run_factory):
        unigram = train_mod.train(run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, ngram_max=1)).ctx)
        bigram = train_mod.train(run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, ngram_max=2)).ctx)
        size = lambda m: len(m.estimator.named_steps["tfidf"].vocabulary_)  # noqa: E731
        assert size(bigram) > size(unigram)

    def test_a_blank_text_is_not_trained_on(self, run_factory):
        frame = text_frame()
        frame.loc[frame.index[:4], "text"] = "   "
        train_mod.train(run_factory(frame, compile_(TEXT, TEXT_TABLE, seed=0)).ctx)


# -- Evaluation ---------------------------------------------------------------------------------------------


class TestEvaluate:
    def test_a_classifier_gets_the_shared_report(self, run_factory):
        frame = table_frame()
        model = train_mod.train(run_factory(frame, compile_(seed=0)).ctx)
        result = SklearnBackend.evaluate(model, frame, "split", C.ITEM_ID_COLUMN_NAME)
        assert result is not None
        report = result.report
        assert report["split"] == "test" and report["outputType"] == "category"
        assert report["idx2str"] == ["high", "low", "mid"]
        assert report["overall"]["accuracy"] > 0.95
        assert np.asarray(report["confusionMatrix"]).sum() == report["rowCount"]
        assert {"class_actual", "class_predictions", C.ITEM_ID_COLUMN_NAME} <= set(result.predictions.columns)

    def test_a_row_with_an_empty_cell_is_still_evaluated_for_a_table_but_not_for_text(self, run_factory):
        frame = table_frame()
        model = train_mod.train(run_factory(frame, compile_(model_id="logistic_regression", seed=0)).ctx)
        test_rows = frame[frame["split"] == "test"]
        frame.loc[test_rows.index[0], "x1"] = np.nan
        result = SklearnBackend.evaluate(model, frame, "split", C.ITEM_ID_COLUMN_NAME)
        assert result is not None and result.report["rowCount"] == len(test_rows)

        text = text_frame()
        text_model = train_mod.train(run_factory(text, compile_(TEXT, TEXT_TABLE, seed=0)).ctx)
        text_test = text[text["split"] == "test"]
        text.loc[text_test.index[0], "text"] = None
        result = SklearnBackend.evaluate(text_model, text, "split", C.ITEM_ID_COLUMN_NAME)
        assert result is not None and result.report["rowCount"] == len(text_test) - 1

    def test_a_regressor_gets_mae_rmse_and_r2(self, run_factory):
        frame = regression_frame()
        model = train_mod.train(run_factory(frame, compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge")).ctx)
        result = SklearnBackend.evaluate(model, frame, "split", C.ITEM_ID_COLUMN_NAME)
        assert result is not None
        overall = result.report["overall"]
        assert result.report["outputType"] == "number" and result.report["split"] == "test"
        assert overall["r2"] > 0.99 and overall["mae"] < 0.5 and overall["rmse"] < 0.5
        assert {"target_actual", "target_predictions"} <= set(result.predictions.columns)

    def test_the_headline_score_is_accuracy_for_a_classifier_and_r2_for_a_regressor(self, run_factory):
        from omnitrain.services.evaluation_score import score_from_overall

        table = table_frame()
        classifier = train_mod.train(run_factory(table, compile_(seed=0)).ctx)
        report = SklearnBackend.evaluate(classifier, table, "split", C.ITEM_ID_COLUMN_NAME).report  # type: ignore[union-attr]
        assert score_from_overall(report["overall"]).metric == "accuracy"  # type: ignore[union-attr]
        numbers = regression_frame()
        regressor = train_mod.train(run_factory(numbers, compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge")).ctx)
        report = SklearnBackend.evaluate(regressor, numbers, "split", C.ITEM_ID_COLUMN_NAME).report  # type: ignore[union-attr]
        assert score_from_overall(report["overall"]).metric == "r2"  # type: ignore[union-attr]

    def test_a_constant_target_has_no_r2(self, run_factory):
        frame = regression_frame()
        model = train_mod.train(run_factory(frame, compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge")).ctx)
        frame["target"] = 5.0
        result = SklearnBackend.evaluate(model, frame, "split", C.ITEM_ID_COLUMN_NAME)
        assert result is not None and result.report["overall"]["r2"] is None


# -- The model file -----------------------------------------------------------------------------------------


class Gadget:
    """Not a scikit-learn type: a stand-in for anything a forged model file might smuggle in."""

    def __init__(self):
        self.x = 1


class TestModelFileSafety:
    def test_a_file_holding_a_foreign_type_is_refused(self, tmp_path):
        import skops.io

        skops.io.dump(Gadget(), str(tmp_path / estimators.MODEL_FILENAME))
        with pytest.raises(Exception, match="Untrusted"):
            estimators.load(str(tmp_path))

    def test_a_pickle_is_not_a_model_file(self, tmp_path):
        import pickle

        (tmp_path / estimators.MODEL_FILENAME).write_bytes(pickle.dumps(Gadget()))
        with pytest.raises(Exception):  # noqa: B017 - what is raised is skops' business; that it is raised is ours
            estimators.load(str(tmp_path))

    def test_only_reviewed_types_are_trusted_and_every_model_loads_with_just_those(self):
        assert estimators.TRUSTED == [
            "numpy.dtype",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "sklearn.tree._tree.Tree",
        ]


# -- Export -------------------------------------------------------------------------------------------------

TEMPLATES = Path(__file__).resolve().parents[1] / "omnitrain" / "plugins" / "export" / "templates" / "python"


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def export(model, tmp_path: Path, run_id: str = "run-1") -> tuple[Path, Path]:
    """Convert a trained model and write its manifest next to it, as a bundle does."""
    path = Path(SklearnBackend.convert(model, "onnx", str(tmp_path / "export")))
    manifest = SklearnBackend.preprocessing_manifest(run_id, {}, model.meta)
    manifest_path = tmp_path / "preprocessing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest_path


def generated_client(model_path: Path, manifest_path: Path):
    """The Python devkit client every export ships, executed from its template source. It is not imported as a
    module: that would write a bytecode cache into the templates folder, which is shipped as is."""
    import types

    source = (TEMPLATES / "omnitrain_client.py").read_text(encoding="utf-8")
    module = types.ModuleType("omnitrain_client_under_test")
    module.__file__ = str(TEMPLATES / "omnitrain_client.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - our own shipped template
    return module.OmniTrainClient(model_path, manifest_path)


class TestExport:
    @pytest.mark.parametrize("model_id", ["hist_gradient_boosting", "random_forest", "logistic_regression"])
    def test_a_classifier_graph_gives_the_same_probabilities_through_the_clients_softmax(
        self, run_factory, tmp_path, model_id
    ):
        import onnxruntime as ort

        frame = table_frame()
        frame.loc[frame.index[7], "x1"] = np.nan  # a missing value, which the graph imputes or handles
        model = train_mod.train(run_factory(frame, compile_(model_id=model_id, seed=0)).ctx)
        path, _ = export(model, tmp_path)

        session = ort.InferenceSession(str(path))
        assert [i.name for i in session.get_inputs()] == ["x1", "x2"]
        assert [o.name for o in session.get_outputs()] == ["class"]
        feed = {c: frame[c].to_numpy(dtype=np.float32) for c in ("x1", "x2")}
        logits = session.run(None, feed)[0]
        expected = np.asarray(model.predict(frame[["x1", "x2"]])["class_probabilities"].tolist())
        assert np.abs(softmax(logits) - expected).max() < 1e-4

    @pytest.mark.parametrize("model_id", ["hist_gradient_boosting", "random_forest", "ridge"])
    def test_a_regressor_graph_gives_the_same_numbers(self, run_factory, tmp_path, model_id):
        import onnxruntime as ort

        frame = regression_frame()
        config = compile_(REGRESSION, REGRESSION_TABLE, model_id=model_id, seed=0)
        model = train_mod.train(run_factory(frame, config).ctx)
        path, _ = export(model, tmp_path)
        session = ort.InferenceSession(str(path))
        assert [o.name for o in session.get_outputs()] == ["target"]
        out = session.run(None, {c: frame[c].to_numpy(dtype=np.float32) for c in ("x1", "x2")})[0]
        expected = model.predict(frame[["x1", "x2"]])["target_predictions"].to_numpy()
        assert np.abs(out.reshape(-1) - expected).max() < 1e-3

    def test_the_generated_python_client_runs_an_exported_classifier(self, run_factory, tmp_path):
        frame = table_frame()
        model = train_mod.train(run_factory(frame, compile_(seed=0)).ctx)
        client = generated_client(*export(model, tmp_path))
        row = frame.iloc[0]
        got = client.predict({"x1": float(row["x1"]), "x2": float(row["x2"])}, top_k=3)
        expected = model.golden_prediction(model.predict(frame.iloc[[0]]))
        assert list(got) == list(expected)
        assert all(abs(got[k] - expected[k]) < 1e-3 for k in expected)

    def test_the_generated_python_client_runs_an_exported_regressor(self, run_factory, tmp_path):
        frame = regression_frame()
        model = train_mod.train(run_factory(frame, compile_(REGRESSION, REGRESSION_TABLE, model_id="ridge")).ctx)
        client = generated_client(*export(model, tmp_path))
        row = frame.iloc[0]
        got = client.predict({"x1": float(row["x1"]), "x2": float(row["x2"])})
        expected = float(model.predict(frame.iloc[[0]])["target_predictions"].iloc[0])
        assert got["value"] == pytest.approx(expected, abs=1e-3)

    def test_the_manifest_asks_for_raw_values_and_lists_the_models_own_classes(self, run_factory, tmp_path):
        model = train_mod.train(run_factory(table_frame(), compile_(seed=0)).ctx)
        _, manifest_path = export(model, tmp_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert [(i["name"], i["type"]) for i in manifest["inputs"]] == [("x1", "number"), ("x2", "number")]
        # The graph scales by itself; a client scaling again would be wrong.
        assert not any("numberNormalization" in i for i in manifest["inputs"])
        assert manifest["outputs"] == [
            {"name": "class", "type": "category", "column": "class", "classes": ["high", "low", "mid"]}
        ]

    def test_the_manifest_needs_the_training_metadata(self):
        with pytest.raises(ValueError, match="names.json"):
            SklearnBackend.preprocessing_manifest("run-1", {}, None)

    def test_a_text_graph_takes_a_raw_string_and_matches_the_model(self, run_factory, tmp_path):
        import onnxruntime as ort

        model = train_mod.train(run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, seed=0)).ctx)
        path, manifest_path = export(model, tmp_path)
        session = ort.InferenceSession(str(path))
        assert [i.name for i in session.get_inputs()] == ["text"]
        texts = ["a great and good film", "an awful bad film", "Really GREAT, fine acting!"]
        logits = session.run(None, {"text": np.array(texts).reshape(-1, 1)})[0]
        expected = np.asarray(model.predict(pd.DataFrame({"text": texts}))["class_probabilities"].tolist())
        assert np.abs(softmax(logits) - expected).max() < 1e-4
        entry = json.loads(manifest_path.read_text(encoding="utf-8"))["inputs"][0]
        assert entry["type"] == "text" and entry["textInput"]["runtime"] == "onnxruntime"

    def test_naive_bayes_text_exports_too(self, run_factory, tmp_path):
        import onnxruntime as ort

        model = train_mod.train(run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, model_id="naive_bayes")).ctx)
        path, _ = export(model, tmp_path)
        out = ort.InferenceSession(str(path)).run(None, {"text": np.array([["great film"]])})[0]
        assert out.shape == (1, 2)

    def test_character_ngrams_cannot_be_exported_and_the_error_says_so(self, run_factory, tmp_path):
        model = train_mod.train(run_factory(text_frame(), compile_(TEXT, TEXT_TABLE, analyzer="char_wb")).ctx)
        with pytest.raises(RuntimeError, match="word analyzer"):
            SklearnBackend.convert(model, "onnx", str(tmp_path))

    def test_a_column_named_like_an_internal_tensor_is_refused_not_miswired(self, run_factory, tmp_path):
        frame = table_frame().rename(columns={"x1": "probabilities"})
        ctx = SnapshotContext(
            columns=cols(("probabilities", "scalar"), ("x2", "scalar"), ("class", "label"), ("split", "split")),
            label_class_names=["low", "mid", "high"],
        )
        model = train_mod.train(run_factory(frame, compile_(ctx=ctx, seed=0)).ctx)
        with pytest.raises(RuntimeError, match="clash"):
            SklearnBackend.convert(model, "onnx", str(tmp_path))

    def test_the_conversion_leaves_onnxs_helper_as_it_found_it(self, run_factory, tmp_path):
        from onnx import helper

        before = helper.make_attribute
        model = train_mod.train(run_factory(table_frame(), compile_(seed=0)).ctx)
        SklearnBackend.convert(model, "onnx", str(tmp_path))
        assert helper.make_attribute is before
