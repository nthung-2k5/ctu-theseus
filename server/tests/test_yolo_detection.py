"""The YOLO backend for object detection: compile, dataset, training glue, prediction, evaluation, manifest and
which export formats are offered. `YOLO` itself is a fake here; test_yolo_smoke.py runs the real library."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from theseus.backends.base import ConfigError, TrainContext
from theseus.backends.yolo import YoloBackend, data, runtime
from theseus.backends.yolo import train as train_mod
from theseus.backends.yolo.compile import YoloHyperparameters, compile_yolo_config, hyperparameter_specs
from theseus.backends.yolo.evaluate import EVALUATION_CONFIDENCE
from theseus.backends.yolo.evaluate import evaluate as evaluate_detection
from theseus.backends.yolo.manifest import build_preprocessing_manifest
from theseus.backends.yolo.model import DETECT_CONFIDENCE, YoloLoadedModel
from theseus.export.registry import list_export_formats
from theseus.jobs import abort
from theseus.services.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

TASK = get_task_descriptor("object_detection")
CLASSES = ["cat", "dog"]
CAT = {"label": "cat", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
DOG = {"label": "dog", "x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5}
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="image_path", kind="storage_uri"),
        ColumnSpec(name="boxes", kind="boxes"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=CLASSES,
)
META = {
    "schemaVersion": 1,
    "task": "detect",
    "imgsz": 640,
    "imageColumn": "image_path",
    "outputName": "boxes",
    "names": CLASSES,
}


def make_config(**sel):
    return compile_yolo_config(TASK, SNAPSHOT, YoloHyperparameters(**sel))


def row(item_id, boxes, split):
    return {
        "image_path": f"s3://theseus-datasets/pool/{item_id}.png",
        "boxes": None if boxes is None else json.dumps(boxes),
        "split": split,
        "_theseus_item_id": item_id,
    }


def make_df():
    return pd.DataFrame(
        [
            row("a", [CAT, DOG], "train"),
            row("b", [], "train"),
            row("c", [DOG], "validation"),
            row("d", [CAT], "test"),
        ]
    )


@pytest.fixture
def downloads(monkeypatch):
    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        Path(local_path).write_bytes(b"png")
        return local_path

    monkeypatch.setattr(data.storage, "download_file", fake_download)


# -- Compile ---------------------------------------------------------------------------------------------


class TestCompile:
    def test_defaults_are_a_detection_model_and_a_larger_image(self):
        config = make_config()
        assert (config["task"], config["model"], config["weights"]) == ("detect", "yolo11n", "yolo11n.pt")
        assert config["train"]["imgsz"] == 640 and config["train"]["epochs"] == 50
        assert config["imageColumn"] == "image_path"
        assert config["labelColumn"] == "boxes"  # the ground truth is the boxes column, not a class column
        assert config["classNames"] == CLASSES

    def test_a_single_class_is_a_valid_detector_but_none_is_not(self):
        compile_yolo_config(
            TASK, SnapshotContext(columns=SNAPSHOT.columns, label_class_names=["cat"]), YoloHyperparameters()
        )
        with pytest.raises(ConfigError, match="at least one class"):
            compile_yolo_config(TASK, SnapshotContext(columns=SNAPSHOT.columns), YoloHyperparameters())

    def test_a_classification_model_is_not_a_detection_model_and_vice_versa(self):
        with pytest.raises(ConfigError, match="yolo11n-cls"):
            make_config(model_id="yolo11n-cls")
        classification = get_task_descriptor("image_classification")
        with pytest.raises(ConfigError, match="yolo11n"):
            compile_yolo_config(classification, SnapshotContext(columns=[
                ColumnSpec(name="image_path", kind="storage_uri"), ColumnSpec(name="class", kind="label")
            ], label_class_names=CLASSES), YoloHyperparameters(model_id="yolo11n"))  # fmt: skip

    def test_a_snapshot_without_a_boxes_column_is_refused(self):
        classification_columns = [
            ColumnSpec(name="image_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
        ]
        with pytest.raises(ConfigError, match="boxes"):
            compile_yolo_config(
                TASK, SnapshotContext(columns=classification_columns, label_class_names=CLASSES), YoloHyperparameters()
            )

    def test_models_and_form_specs_exist_for_detection(self):
        assert [m.id for m in YoloBackend.models(TASK)][:2] == ["yolo11n", "yolo11s"]
        specs = {s.name: s for s in hyperparameter_specs(TASK)}
        assert specs["imageSize"].default == "640" and specs["epochs"].default == 50
        YoloHyperparameters.model_validate({name: s.default for name, s in specs.items()})


# -- Dataset ---------------------------------------------------------------------------------------------


def materialize(df, dest, class_names=tuple(CLASSES), **kw):
    return data.materialize_detection(
        df,
        str(dest),
        image_column="image_path",
        boxes_column="boxes",
        split_column="split",
        item_id_column="_theseus_item_id",
        class_names=list(class_names),
        **kw,
    )


def files(root) -> list[str]:
    return sorted(
        os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, fs in os.walk(root) for f in fs
    )


class TestDataset:
    def test_images_and_labels_live_in_parallel_trees(self, tmp_path, downloads):
        result = materialize(make_df(), tmp_path)

        assert files(tmp_path) == [
            "data.yaml",
            "images/test/d.png",
            "images/train/a.png",
            "images/train/b.png",
            "images/val/c.png",
            "labels/test/d.txt",
            "labels/train/a.txt",
            "labels/train/b.txt",
            "labels/val/c.txt",
        ]
        assert result.counts == {"train": 2, "val": 1, "test": 1}
        assert result.box_counts == {"train": 2, "val": 1, "test": 1}
        assert result.data_yaml == str(tmp_path / "data.yaml")

    def test_label_lines_are_class_then_centre_and_size(self, tmp_path, downloads):
        materialize(make_df(), tmp_path)
        lines = (tmp_path / "labels" / "train" / "a.txt").read_text().splitlines()
        # Stored as the top-left corner and size, so the cat box (0.1, 0.2, 0.3, 0.4) is centred at (0.25, 0.4).
        assert lines == ["0 0.250000 0.400000 0.300000 0.400000", "1 0.750000 0.750000 0.500000 0.500000"]

    def test_an_image_with_no_objects_gets_an_empty_label_file(self, tmp_path, downloads):
        materialize(make_df(), tmp_path)
        # Present-but-empty is what tells Ultralytics "background image"; a missing file would mean "unlabelled".
        assert (tmp_path / "labels" / "train" / "b.txt").read_text() == ""

    def test_data_yaml_lists_the_real_class_names_in_index_order(self, tmp_path, downloads):
        names = ("red square", "green: square/odd", "ünïcode")
        df = pd.DataFrame(
            [row("a", [{"label": names[2], "x": 0, "y": 0, "w": 1, "h": 1}], "train"), row("v", [], "validation")]
        )
        materialize(df, tmp_path, class_names=names)

        config = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
        assert config["names"] == {0: names[0], 1: names[1], 2: names[2]}
        assert (config["train"], config["val"]) == ("images/train", "images/val")
        assert "test" not in config  # no test images: Ultralytics rejects a key whose folder holds nothing
        assert os.path.isabs(config["path"])

    def test_unlabelled_images_and_boxes_of_unknown_classes_are_left_out(self, tmp_path, downloads):
        df = pd.DataFrame(
            [
                row("ok", [CAT], "train"),
                row("v", [], "validation"),
                row("unlabelled", None, "train"),  # no `boxes` value: nobody labelled it
                row("alien", [{**CAT, "label": "fish"}], "train"),  # a class the snapshot does not list
                row("nosplit", [CAT], "holdout"),
            ]
        )
        result = materialize(df, tmp_path)
        assert result.skipped == 3
        assert sorted(os.listdir(tmp_path / "labels" / "train")) == ["ok.txt"]

    def test_needs_images_in_train_and_validation_and_boxes_in_train(self, tmp_path, downloads):
        with pytest.raises(ValueError, match="no usable images"):
            materialize(pd.DataFrame([row("a", [CAT], "train")]), tmp_path / "a")
        with pytest.raises(ValueError, match="no boxes"):
            materialize(pd.DataFrame([row("a", [], "train"), row("v", [CAT], "validation")]), tmp_path / "b")

    def test_evaluation_only_puts_every_row_in_val_and_points_train_at_it(self, tmp_path, downloads):
        df = pd.DataFrame([row("t1", [CAT], "test"), row("t2", [], "test")])
        result = materialize(df, tmp_path, evaluation_only=True)

        assert result.counts["val"] == 2 and result.counts["train"] == 0
        config = yaml.safe_load((tmp_path / "data.yaml").read_text())
        assert (config["train"], config["val"]) == ("images/val", "images/val")

    def test_evaluation_only_still_needs_one_image_to_arrive(self, tmp_path, monkeypatch):
        def broken(bucket, key, local_path):
            raise OSError("down")

        monkeypatch.setattr(data.storage, "download_file", broken)
        with pytest.raises(ValueError, match="evaluation images"):
            materialize(pd.DataFrame([row("t1", [CAT], "test")]), tmp_path, evaluation_only=True)

    def test_label_lines_clamp_to_the_image_and_refuse_unknown_classes(self):
        index = {"cat": 0}
        assert data.label_lines([{"label": "cat", "x": 0.9, "y": 0.0, "w": 0.1000001, "h": 1.0}], index) == [
            "0 0.950000 0.500000 0.100000 1.000000"
        ]
        assert data.label_lines([{"label": "fish", "x": 0, "y": 0, "w": 1, "h": 1}], index) is None
        assert data.label_lines([], index) == []


# -- Training --------------------------------------------------------------------------------------------


class FakeDetectYOLO:
    """Stands in for ultralytics.YOLO: leaves the files a real run leaves and fires its callbacks."""

    instances: list["FakeDetectYOLO"] = []
    names: dict[int, str] = {0: "cat", 1: "dog"}

    def __init__(self, weights):
        self.weights = weights
        self.callbacks: dict[str, list] = {}
        self.kwargs: dict = {}
        self.names = dict(FakeDetectYOLO.names)
        FakeDetectYOLO.instances.append(self)

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def fire(self, event, trainer):
        for fn in self.callbacks.get(event, []):
            fn(trainer)

    def train(self, **kwargs):
        self.kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"weights")
        (weights.parent / "results.csv").write_text("epoch\n")
        (weights.parent / "args.yaml").write_text("a: 1\n")
        for epoch in range(2):
            trainer = SimpleNamespace(
                epoch=epoch,
                metrics={
                    "val/box_loss": 1.0,
                    "val/cls_loss": 0.5,
                    "val/dfl_loss": 0.25,
                    "metrics/precision(B)": 0.6,
                    "metrics/recall(B)": 0.7,
                    "metrics/mAP50(B)": 0.8,
                    "metrics/mAP50-95(B)": 0.4,
                },
                tloss=SimpleNamespace(box=2.0, cls=1.0, dfl=0.5),
                label_loss_items=lambda t: {"train/box_loss": t.box, "train/cls_loss": t.cls, "train/dfl_loss": t.dfl},
            )
            self.fire("on_train_epoch_start", trainer)
            self.fire("on_train_batch_end", trainer)
            self.fire("on_fit_epoch_end", trainer)
        # Fired once more after the final validation: only the mAP-style metrics, no losses.
        self.fire(
            "on_fit_epoch_end",
            SimpleNamespace(epoch=2, metrics={"metrics/mAP50(B)": 0.8}, tloss=None, label_loss_items=None),
        )


@pytest.fixture
def train_env(monkeypatch, tmp_path, downloads):
    FakeDetectYOLO.instances = []
    FakeDetectYOLO.names = {0: "cat", 1: "dog"}
    uploads: list[dict] = []
    monkeypatch.setattr(train_mod, "_yolo_class", lambda: FakeDetectYOLO)
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: make_df())
    monkeypatch.setattr(runtime, "device", lambda: "cpu")
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    def fake_upload_dir(bucket, prefix, local_dir, **kw):
        uploads.append(
            {"files": sorted(os.listdir(local_dir)), "meta": json.loads(Path(local_dir, "names.json").read_text())}
        )

    monkeypatch.setattr(train_mod.storage, "upload_dir", fake_upload_dir)
    monkeypatch.setattr(train_mod, "load_model", lambda d: SimpleNamespace(loaded_from=d))
    return SimpleNamespace(uploads=uploads, workdir=tmp_path / "work")


def make_ctx(env, abort_after=None):
    calls = SimpleNamespace(aborts=0, reports=[])

    def check_abort():
        calls.aborts += 1
        if abort_after is not None and calls.aborts >= abort_after:
            raise abort.TrainingAborted("stop")

    ctx = TrainContext(
        run_id="run-1",
        config=make_config(),
        dataset_uri="s3://theseus-datasets/snapshots/v/dataset.parquet",
        output_uri="s3://theseus-training/run-1/results/",
        workdir=str(env.workdir),
        _heartbeat=lambda: None,
        _check_abort=check_abort,
        _report=lambda epoch, split, metrics: calls.reports.append((epoch, split, metrics)),
    )
    return ctx, calls


class TestTraining:
    def test_trains_from_the_data_yaml_and_reports_summed_losses_with_the_detection_metrics(self, train_env):
        ctx, calls = make_ctx(train_env)
        train_mod.train(ctx)

        kw = FakeDetectYOLO.instances[0].kwargs
        assert kw["data"].endswith("data.yaml") and kw["data"].startswith(str(train_env.workdir))
        assert FakeDetectYOLO.instances[0].weights.endswith("yolo11n.pt")
        assert (kw["imgsz"], kw["epochs"], kw["plots"]) == (640, 50, False)

        validation = {
            "loss": 1.75,
            "box_loss": 1.0,
            "cls_loss": 0.5,
            "dfl_loss": 0.25,
            "precision": 0.6,
            "recall": 0.7,
            "mAP50": 0.8,
            "mAP50-95": 0.4,
        }
        # 1-based epochs; the extra post-validation callback (no losses) is not an epoch.
        assert calls.reports == [
            (1, "train", {"loss": 3.5, "box_loss": 2.0, "cls_loss": 1.0, "dfl_loss": 0.5}),
            (1, "validation", validation),
            (2, "train", {"loss": 3.5, "box_loss": 2.0, "cls_loss": 1.0, "dfl_loss": 0.5}),
            (2, "validation", validation),
        ]

    def test_the_run_uploads_weights_and_names_with_the_detection_task(self, train_env):
        ctx, _ = make_ctx(train_env)
        train_mod.train(ctx)

        (upload,) = train_env.uploads
        assert upload["files"] == ["args.yaml", "best.pt", "names.json", "results.csv"]
        meta = upload["meta"]
        assert (meta["task"], meta["imgsz"], meta["outputName"], meta["imageColumn"]) == (
            "detect",
            640,
            "boxes",
            "image_path",
        )
        assert meta["names"] == ["cat", "dog"]

    def test_names_come_from_the_trained_model_not_from_the_snapshot(self, train_env):
        FakeDetectYOLO.names = {0: "dog", 1: "cat"}  # a model whose own order differs from the compiled config's
        ctx, _ = make_ctx(train_env)
        train_mod.train(ctx)
        assert train_env.uploads[0]["meta"]["names"] == ["dog", "cat"]

    def test_a_class_count_mismatch_is_an_error(self, train_env):
        FakeDetectYOLO.names = {0: "cat"}
        ctx, _ = make_ctx(train_env)
        with pytest.raises(RuntimeError, match="1 classes but the snapshot has 2"):
            train_mod.train(ctx)

    def test_an_abort_stops_training_and_cleans_up(self, train_env):
        ctx, _ = make_ctx(train_env, abort_after=8)
        with pytest.raises(abort.TrainingAborted):
            train_mod.train(ctx)
        assert train_env.uploads == [] and not train_env.workdir.exists()

    def test_validation_metrics_need_all_three_losses(self):
        full = {"val/box_loss": 1.0, "val/cls_loss": 2.0, "val/dfl_loss": 3.0}
        assert train_mod.validation_metrics("detect", full)["loss"] == 6.0
        assert train_mod.validation_metrics("detect", {**full, "val/dfl_loss": None}) is None
        assert train_mod.validation_metrics("detect", {"metrics/mAP50(B)": 0.5}) is None
        # Classification is unchanged.
        assert train_mod.validation_metrics("classify", {"val/loss": 0.3, "metrics/accuracy_top1": 0.9}) == {
            "loss": 0.3,
            "accuracy": 0.9,
        }
        assert train_mod.validation_metrics("classify", {"metrics/accuracy_top1": 0.9}) is None


# -- Prediction ------------------------------------------------------------------------------------------


class FakeBoxes:
    def __init__(self, xyxyn, conf, cls):
        self.xyxyn, self.conf, self.cls = np.array(xyxyn, dtype=float).reshape(-1, 4), np.array(conf), np.array(cls)


class FakePredictYOLO:
    """Returns scripted detections per source path, and scripted validator metrics from `.val`."""

    def __init__(self, scripted, val_metrics=None):
        self.scripted, self.val_metrics = scripted, val_metrics
        self.predict_calls: list[dict] = []
        self.val_calls: list[dict] = []

    def predict(self, sources, **kwargs):
        self.predict_calls.append({"sources": list(sources), **kwargs})
        return [
            SimpleNamespace(orig_shape=(96, 128), boxes=FakeBoxes(*self.scripted.get(s, ([], [], [])))) for s in sources
        ]

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        assert os.path.isfile(kwargs["data"]), "the evaluation dataset must exist while the validator runs"
        return self.val_metrics


@pytest.fixture(autouse=False)
def cpu(monkeypatch):
    monkeypatch.setattr(runtime, "device", lambda: "cpu")
    monkeypatch.setattr(runtime, "dataloader_workers", lambda: 0)


SCRIPT = {
    # Two detections, deliberately out of confidence order, one box slightly outside the image.
    "a.png": ([[0.1, 0.2, 0.4, 0.6], [0.5, 0.5, 1.02, 1.0]], [0.4, 0.9], [0, 1]),
    "empty.png": ([], [], []),
}


class TestPrediction:
    def test_detections_are_boxes_in_the_annotation_convention_most_confident_first(self, cpu):
        model = YoloLoadedModel(FakePredictYOLO(SCRIPT), META)
        frame = model.predict(pd.DataFrame({"image_path": ["a.png"]}))

        detections = frame["boxes_predictions"].iloc[0]
        assert [d["label"] for d in detections] == ["dog", "cat"]
        cat = detections[1]["box"]
        assert (cat["x"], cat["y"]) == (0.1, 0.2)
        assert cat["w"] == pytest.approx(0.3) and cat["h"] == pytest.approx(0.4)  # corner + size, not corner + corner
        dog = detections[0]["box"]
        assert dog["x"] + dog["w"] == pytest.approx(1.0)  # clamped to the image edge
        assert list(frame["boxes_image_size"].iloc[0]) == [128, 96]

    def test_serving_uses_the_serving_threshold_and_evaluation_can_lower_it(self, cpu):
        yolo = FakePredictYOLO(SCRIPT)
        model = YoloLoadedModel(yolo, META)
        frame = pd.DataFrame({"image_path": ["a.png"]})
        model.predict(frame)
        model.predict(frame, confidence=EVALUATION_CONFIDENCE)
        assert [c["conf"] for c in yolo.predict_calls] == [DETECT_CONFIDENCE, EVALUATION_CONFIDENCE]
        assert yolo.predict_calls[0]["imgsz"] == 640

    def test_to_output_is_the_detections_shape(self, cpu):
        model = YoloLoadedModel(FakePredictYOLO(SCRIPT), META)
        out = model.to_output(model.predict(pd.DataFrame({"image_path": ["a.png"]})))

        assert out["kind"] == "detections"
        assert (out["feature"], out["imageWidth"], out["imageHeight"]) == ("boxes", 128, 96)
        assert [d["label"] for d in out["detections"]] == ["dog", "cat"]
        assert out["detections"][1] == {
            "label": "cat",
            "confidence": 0.4,
            "box": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        }

    def test_top_k_keeps_the_most_confident(self, cpu):
        model = YoloLoadedModel(FakePredictYOLO(SCRIPT), META)
        out = model.to_output(model.predict(pd.DataFrame({"image_path": ["a.png"]})), top_k=1)
        assert [d["label"] for d in out["detections"]] == ["dog"]

    def test_an_image_with_nothing_in_it_is_an_empty_list_not_an_error(self, cpu):
        model = YoloLoadedModel(FakePredictYOLO(SCRIPT), META)
        frame = model.predict(pd.DataFrame({"image_path": ["empty.png"]}))
        out = model.to_output(frame)
        assert out["detections"] == [] and model.golden_prediction(frame) == {}

    def test_golden_prediction_is_the_best_confidence_per_class(self, cpu):
        scripted = {"a.png": ([[0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 0.5, 1]], [0.3, 0.8, 0.6], [0, 0, 1])}
        model = YoloLoadedModel(FakePredictYOLO(scripted), META)
        frame = model.predict(pd.DataFrame({"image_path": ["a.png"]}))

        assert model.golden_prediction(frame) == {"cat": 0.8, "dog": 0.6}
        assert model.golden_prediction(frame, threshold=0.7) == {"cat": 0.8}

    def test_the_output_declares_detections_with_the_model_labels(self):
        model = YoloLoadedModel(FakePredictYOLO({}), META)
        assert (model.output.name, model.output.kind, model.output.labels) == ("boxes", "detections", CLASSES)
        assert model.input_columns == ["image_path"]


# -- Evaluation ------------------------------------------------------------------------------------------


def validator_result(present=(0,), p=(0.5,), r=(0.25,), ap50=(0.4,), ap=(0.2,), instances=(3, 0)):
    return SimpleNamespace(
        box=SimpleNamespace(
            ap_class_index=np.array(present),
            p=np.array(p),
            r=np.array(r),
            ap50=np.array(ap50),
            ap=np.array(ap),
            mp=0.5,
            mr=0.25,
            map50=0.4,
            map=0.2,
        ),
        nt_per_class=np.array(instances),
    )


class TestEvaluation:
    @staticmethod
    def df(split="test"):
        return pd.DataFrame(
            [
                row("a", [CAT, DOG], split),
                row("b", [], split),
                row("unlabelled", None, split),
            ]
        ).assign(image_path=lambda d: ["a.png", "empty.png", "unlabelled.png"])

    def run(self, df, yolo=None, downloads_ok=True):
        yolo = yolo or FakePredictYOLO(SCRIPT, validator_result())
        model = YoloLoadedModel(yolo, META)
        return yolo, evaluate_detection(model, df, "split", "_theseus_item_id")

    def test_the_report_has_the_detection_shape_and_names_every_class(self, cpu, downloads):
        _, result = self.run(self.df())
        assert result is not None
        report = result.report

        assert report["kind"] == "detection" and report["split"] == "test"
        assert report["overall"] == {"precision": 0.5, "recall": 0.25, "map50": 0.4, "map5095": 0.2}
        assert (report["imageCount"], report["instanceCount"]) == (2, 3)  # the unlabelled image is not scored
        assert report["idx2str"] == CLASSES
        # "dog" never occurs in the images, so the validator does not list it: it still gets a row, of zeros.
        assert report["perClass"]["cat"] == {
            "precision": 0.5,
            "recall": 0.25,
            "ap50": 0.4,
            "ap5095": 0.2,
            "instances": 3,
        }
        assert report["perClass"]["dog"] == {
            "precision": 0.0,
            "recall": 0.0,
            "ap50": 0.0,
            "ap5095": 0.0,
            "instances": 0,
        }
        assert "accuracy" not in report["overall"]  # run_evaluations stores accuracy as null for a detector

    def test_the_validator_runs_at_a_very_low_confidence_on_a_dataset_that_exists_for_that_call(self, cpu, downloads):
        yolo, _ = self.run(self.df())
        (call,) = yolo.val_calls
        assert call["conf"] == EVALUATION_CONFIDENCE and call["split"] == "val" and call["imgsz"] == 640
        assert call["plots"] is False and call["data"].endswith("data.yaml")
        assert not os.path.exists(call["data"])  # and the temporary dataset is gone afterwards

    def test_the_download_pairs_each_image_with_what_was_labelled_and_what_was_predicted(self, cpu, downloads):
        _, result = self.run(self.df())
        assert result is not None
        download = result.predictions
        assert list(download["_theseus_item_id"]) == ["a", "b"]
        assert json.loads(download["boxes_actual"].iloc[0]) == [CAT, DOG]
        predicted = json.loads(download["boxes_predictions"].iloc[0])
        assert [d["label"] for d in predicted] == ["dog", "cat"]
        assert json.loads(download["boxes_predictions"].iloc[1]) == []

    def test_evaluates_test_then_validation_then_everything(self, cpu, downloads):
        mixed = pd.concat([self.df("train"), self.df("validation")], ignore_index=True)
        assert self.run(mixed)[1].report["split"] == "validation"  # type: ignore[union-attr]
        assert self.run(self.df("train"))[1].report["split"] == "full"  # type: ignore[union-attr]

    def test_nothing_labelled_means_no_report(self, cpu, downloads):
        unlabelled = self.df().iloc[2:]
        assert self.run(unlabelled)[1] is None
        assert self.run(self.df().iloc[:0])[1] is None


# -- Manifest and export ---------------------------------------------------------------------------------


class TestManifestAndExport:
    def test_the_detection_manifest_describes_letterboxing_and_the_raw_output(self):
        manifest = build_preprocessing_manifest("run-1", {}, META)
        (entry,) = manifest["inputs"]
        (output,) = manifest["outputs"]

        assert entry["imageResize"]["mode"] == "letterbox" and entry["imageResize"]["padValue"] == 114
        assert entry["imageResize"]["size"] == [640, 640]
        assert entry["ludwigPreprocessing"] == {"height": 640, "width": 640}
        assert (output["type"], output["name"], output["column"]) == ("detections", "boxes", "boxes")
        # The class list is what becomes labels.txt in a bundle, in the model's own order.
        assert output["classes"] == CLASSES
        assert output["yoloOutput"]["nonMaxSuppression"] == "required"

    def test_only_the_plain_onnx_format_is_offered_for_detection(self):
        ids = [f.id for f in list_export_formats(TASK, YoloBackend)]
        assert ids == ["onnx"]  # no devkit or app can run a detector, and torch_export is not a YOLO artifact

    def test_classification_still_offers_the_devkits_and_apps(self):
        ids = [f.id for f in list_export_formats(get_task_descriptor("image_classification"), YoloBackend)]
        assert {"onnx", "python_devkit", "typescript_devkit", "csharp_devkit", "java_devkit"} <= set(ids)
        assert {"pwa_app", "flutter_app"} <= set(ids)

    def test_the_golden_sample_is_skipped_for_a_detector(self, monkeypatch):
        from theseus.jobs import export as export_job

        class Detector:
            output = SimpleNamespace(kind="detections")

        monkeypatch.setattr(YoloBackend, "load", classmethod(lambda cls, model_dir: Detector()))
        assert export_job._build_golden_sample(YoloBackend, "dir", "key") is None  # type: ignore[arg-type]
