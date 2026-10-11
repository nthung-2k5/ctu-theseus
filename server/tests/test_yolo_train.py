"""YOLO training glue: callbacks translating Ultralytics events into `TrainContext` calls, the local dataset
and the upload. `YOLO` itself is replaced by a fake that fires the same callback events a real trainer does
(the real library is exercised by test_yolo_smoke.py, which is opt-in because it trains for real)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from omnitrain import constants as C
from omnitrain.backends.base import TrainContext
from omnitrain.backends.yolo import data, runtime
from omnitrain.backends.yolo import train as train_mod
from omnitrain.backends.yolo.compile import YoloHyperparameters, compile_yolo_config
from omnitrain.jobs import abort
from omnitrain.services.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

CLASSES = ["cat", "dog", "bird"]


def make_config(**sel):
    ctx = SnapshotContext(
        columns=[
            ColumnSpec(name="image_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=CLASSES,
    )
    return compile_yolo_config(get_task_descriptor("image_classification"), ctx, YoloHyperparameters(**sel))


def make_df():
    rows = [("a", "cat", "train"), ("b", "dog", "train"), ("c", "bird", "train"), ("d", "cat", "validation")]
    return pd.DataFrame(
        [
            {"image_path": f"s3://theseus-datasets/pool/{i}.png", "class": c, "split": s, "_theseus_item_id": i}
            for i, c, s in rows
        ]
    )


class FakeYOLO:
    """Stands in for ultralytics.YOLO: writes the files a real run leaves behind and fires its callbacks."""

    instances: list["FakeYOLO"] = []
    epochs = 2
    # What Ultralytics calls the classes: the folder names data.py wrote, in the order it sorted them.
    names = {0: "0", 1: "1", 2: "2"}
    write_best = True

    def __init__(self, weights):
        self.weights = weights
        self.callbacks: dict[str, list] = {}
        self.kwargs: dict = {}
        self.names = dict(FakeYOLO.names)
        FakeYOLO.instances.append(self)

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def fire(self, event, trainer):
        for fn in self.callbacks.get(event, []):
            fn(trainer)

    def train(self, **kwargs):
        self.kwargs = kwargs
        weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights.mkdir(parents=True)
        (weights / ("best.pt" if self.write_best else "last.pt")).write_bytes(b"weights")
        (weights.parent / "results.csv").write_text("epoch\n")
        (weights.parent / "args.yaml").write_text("a: 1\n")
        (weights.parent / "plots.png").write_bytes(b"x")  # must not be uploaded
        self.fire("on_train_start", SimpleNamespace())
        for epoch in range(self.epochs):
            trainer = SimpleNamespace(
                epoch=epoch,
                metrics={
                    "metrics/accuracy_top1": 0.5 + epoch / 10,
                    "metrics/accuracy_top5": 1.0,
                    "val/loss": 0.9 - epoch,
                },
                tloss=SimpleNamespace(loss=1.0 + epoch),
                label_loss_items=lambda tloss: {"train/loss": tloss.loss},
            )
            self.fire("on_train_epoch_start", trainer)
            self.fire("on_train_batch_end", trainer)
            self.fire("on_fit_epoch_end", trainer)
        # Ultralytics fires this once more after the final validation: no val/loss, same epoch number again.
        self.fire(
            "on_fit_epoch_end",
            SimpleNamespace(
                epoch=self.epochs, metrics={"metrics/accuracy_top1": 0.7}, tloss=None, label_loss_items=None
            ),
        )


@pytest.fixture
def env(monkeypatch, tmp_path):
    FakeYOLO.instances, FakeYOLO.epochs, FakeYOLO.write_best = [], 2, True
    uploads: list[dict] = []
    monkeypatch.setattr(train_mod, "_yolo_class", lambda: FakeYOLO)
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: make_df())
    monkeypatch.setattr(runtime, "device", lambda: "cpu")
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        Path(local_path).write_bytes(b"png")
        return local_path

    monkeypatch.setattr(data.storage, "download_file", fake_download)

    def fake_upload_dir(bucket, prefix, local_dir, **kw):
        uploads.append(
            {
                "bucket": bucket,
                "prefix": prefix,
                "files": sorted(os.listdir(local_dir)),
                "names": json.loads(Path(local_dir, "names.json").read_text()),
            }
        )
        return len(uploads[-1]["files"])

    monkeypatch.setattr(train_mod.storage, "upload_dir", fake_upload_dir)
    loaded: list[str] = []
    monkeypatch.setattr(train_mod, "load_model", lambda d: loaded.append(d) or SimpleNamespace(loaded_from=d))
    return SimpleNamespace(uploads=uploads, loaded=loaded, workdir=tmp_path / "work")


def make_ctx(env, config=None, abort_after=None):
    calls = SimpleNamespace(heartbeats=0, aborts=0, reports=[])

    def check_abort():
        calls.aborts += 1
        if abort_after is not None and calls.aborts >= abort_after:
            raise abort.TrainingAborted("stop")

    def heartbeat():
        calls.heartbeats += 1

    ctx = TrainContext(
        run_id="run-1",
        config=config or make_config(),
        dataset_uri="s3://theseus-datasets/snapshots/v/dataset.parquet",
        output_uri="s3://theseus-training/run-1/results/",
        workdir=str(env.workdir),
        _heartbeat=heartbeat,
        _check_abort=check_abort,
        _report=lambda epoch, split, metrics: calls.reports.append((epoch, split, metrics)),
    )
    return ctx, calls


def test_trains_on_a_local_dataset_inside_the_workdir_and_reports_per_epoch(env):
    ctx, calls = make_ctx(env)
    train_mod.train(ctx)

    fake = FakeYOLO.instances[0]
    kw = fake.kwargs
    assert fake.weights.endswith("yolo11n-cls.pt")
    assert os.path.isabs(fake.weights)  # a bare name would be downloaded into the current directory
    assert kw["project"].startswith(str(env.workdir)) and kw["name"] == "run" and kw["exist_ok"] is True
    assert kw["data"].startswith(str(env.workdir))
    assert (kw["plots"], kw["verbose"], kw["device"]) == (False, False, "cpu")
    assert kw["epochs"] == 30 and kw["imgsz"] == 224 and kw["optimizer"] == "AdamW"

    # Epoch numbers are 1-based; the extra post-validation callback is not reported as an epoch.
    assert calls.reports == [
        (1, "train", {"loss": 1.0}),
        (1, "validation", {"loss": 0.9, "accuracy": 0.5, "top5Accuracy": 1.0}),
        (2, "train", {"loss": 2.0}),
        (2, "validation", {"loss": pytest.approx(-0.1), "accuracy": pytest.approx(0.6), "top5Accuracy": 1.0}),
    ]


def test_each_epoch_checks_for_abort_and_beats(env):
    ctx, calls = make_ctx(env)
    train_mod.train(ctx)
    assert calls.heartbeats >= 3  # after reading the parquet and at each epoch start, at least
    assert calls.aborts >= 4  # downloads, batch ends, epoch starts and ends


def test_an_abort_raised_in_a_callback_stops_training_and_cleans_up(env):
    # 4 checks for the image downloads and 1 after them; the 8th lands in the first epoch's end-of-epoch callback.
    ctx, _ = make_ctx(env, abort_after=8)
    with pytest.raises(abort.TrainingAborted):
        train_mod.train(ctx)
    assert env.uploads == []
    assert not env.workdir.exists()


def test_abort_during_the_dataset_download_never_starts_training(env):
    ctx, _ = make_ctx(env, abort_after=1)
    with pytest.raises(abort.TrainingAborted):
        train_mod.train(ctx)
    assert FakeYOLO.instances == []


def test_uploads_the_best_weights_names_and_metrics_but_not_plots_or_last_pt(env):
    ctx, _ = make_ctx(env)
    train_mod.train(ctx)

    (upload,) = env.uploads
    assert upload["bucket"] == C.BUCKET_TRAINING
    assert upload["prefix"] == "run-1/results/"
    assert upload["files"] == ["args.yaml", "best.pt", "names.json", "results.csv"]


def test_names_json_is_derived_from_the_trained_models_class_order(env):
    # The folder names are indices into the snapshot class list; a model whose own order differs from the
    # snapshot's (here reversed) must come out labelled by ITS order, not by the snapshot's.
    FakeYOLO.names = {0: "2", 1: "1", 2: "0"}
    ctx, _ = make_ctx(env)
    train_mod.train(ctx)

    names = env.uploads[0]["names"]
    assert names["names"] == ["bird", "dog", "cat"]
    assert (names["imgsz"], names["imageColumn"], names["outputName"], names["task"]) == (
        224,
        "image_path",
        "class",
        "classify",
    )


def test_the_returned_model_is_loaded_from_what_was_uploaded(env):
    ctx, _ = make_ctx(env)
    result = train_mod.train(ctx)
    assert len(env.loaded) == 1 and env.loaded[0].endswith("upload")
    assert result.loaded_from == env.loaded[0]  # type: ignore[attr-defined]


def test_falls_back_to_last_pt_when_no_best_checkpoint_was_written(env):
    FakeYOLO.write_best = False
    ctx, _ = make_ctx(env)
    train_mod.train(ctx)
    assert "best.pt" in env.uploads[0]["files"]


def test_the_workdir_is_removed_after_success_and_after_failure(env, monkeypatch):
    ctx, _ = make_ctx(env)
    train_mod.train(ctx)
    assert not env.workdir.exists()

    monkeypatch.setattr(FakeYOLO, "train", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        train_mod.train(make_ctx(env)[0])
    assert not env.workdir.exists()


def test_class_count_mismatch_is_an_error_not_a_silent_mislabel(env):
    FakeYOLO.names = {0: "0", 1: "1"}
    ctx, _ = make_ctx(env)
    with pytest.raises(RuntimeError, match="2 classes but the snapshot has 3"):
        train_mod.train(ctx)


def test_heartbeats_are_throttled(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(train_mod.time, "monotonic", lambda: now["t"])
    beats = []
    ctx = TrainContext("r", {}, "d", "o", "w", lambda: beats.append(1), lambda: None, lambda *a: None)
    beat = train_mod._Beat(ctx)
    for _ in range(500):
        beat()
    assert len(beats) == 1
    now["t"] += train_mod.HEARTBEAT_EVERY_SECONDS
    beat()
    assert len(beats) == 2


def test_compiled_train_options_never_clash_with_the_kwargs_train_sets_itself():
    clash = {"data", "project", "name", "exist_ok", "plots", "verbose", "workers", "device"}
    assert not clash & set(make_config(freeze_layers=2, seed=1)["train"])
