"""The YOLO half of training: materialise the snapshot locally, run `YOLO.train()`, stream progress, upload.

Differences from the Ludwig backend that shape this module:

  * Ultralytics reads and writes local files only, so the dataset is downloaded into `run.workdir`
    (`data.py`) and the results are uploaded afterwards. `run.output_uri` is not used for writing.
  * Progress, heartbeats and the abort check are Ultralytics callbacks. They run synchronously in the
    training thread and an exception raised in one propagates out of `YOLO.train()` untouched, so
    `TrainingAborted` from `run.check_abort()` stops training the same way it does for Ludwig.
  * The class order of a trained model is whatever `model.names` says, so `names.json` is written from
    it and never from the snapshot's class list (see CLAUDE.md, "Class list correctness").
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
from opentelemetry import trace

from omnitrain import constants as C
from omnitrain.backends.base import TrainContext
from omnitrain.backends.yolo import data, runtime
from omnitrain.backends.yolo.model import YoloLoadedModel
from omnitrain.backends.yolo.model import load as load_model
from omnitrain.services import storage

tracer = trace.get_tracer("omnitrain")

# The run is presumed hung when it goes `run_heartbeat_timeout_seconds` (15 min) without a heartbeat, which a long
# epoch on a large dataset can exceed. Beat from the finer hooks too, this often.
HEARTBEAT_EVERY_SECONDS = 10.0

# Ultralytics' files worth keeping. `last.pt` is left out: serving only needs the best weights and it doubles the
# download. Plots are switched off, so there are none.
_UPLOADED = ("results.csv", "args.yaml")

logger = logging.getLogger(__name__)


def _yolo_class() -> Any:
    """`ultralytics.YOLO`, imported lazily and only after `runtime.configure()` has pointed Ultralytics' files
    away from the user's home and current directory. A function so tests can substitute a fake."""
    runtime.configure()
    from ultralytics import YOLO

    return YOLO


class _Beat:
    """Heartbeat at most every HEARTBEAT_EVERY_SECONDS, so calling it per batch stays cheap."""

    def __init__(self, run: TrainContext) -> None:
        self._run = run
        self._last = 0.0

    def __call__(self) -> None:
        now = time.monotonic()
        if now - self._last >= HEARTBEAT_EVERY_SECONDS:
            self._last = now
            self._run.heartbeat()


_DETECT_LOSSES = ("box", "cls", "dfl")
# Ultralytics' detection metric key -> the name stored for it.
_DETECT_VALIDATION_METRICS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
_CLASSIFY_VALIDATION_METRICS = {"metrics/accuracy_top1": "accuracy", "metrics/accuracy_top5": "top5Accuracy"}


def validation_metrics(kind: str, metrics: dict[str, Any]) -> dict[str, float] | None:
    """The metrics to store for the validation split, or None when this callback is not an epoch's end.

    `loss` is always present, since the best epoch is the one with the lowest validation `loss` (events/writer.py).
    Detection has three loss terms rather than one, so `loss` is their sum, each also kept under its own name.
    """
    if kind == "detect":
        parts = {name: metrics.get(f"val/{name}_loss") for name in _DETECT_LOSSES}
        if any(v is None for v in parts.values()):
            return None
        out = {"loss": sum(parts.values()), **{f"{name}_loss": v for name, v in parts.items()}}
        wanted = _DETECT_VALIDATION_METRICS
    else:
        if "val/loss" not in metrics:
            return None
        out = {"loss": metrics["val/loss"]}
        wanted = _CLASSIFY_VALIDATION_METRICS
    out.update({name: metrics[source] for source, name in wanted.items() if source in metrics})
    return {k: float(v) for k, v in out.items()}


def train_metrics(kind: str, trainer: Any) -> dict[str, float] | None:
    if trainer.tloss is None:
        return None
    items = trainer.label_loss_items(trainer.tloss)
    if kind == "detect":
        parts = {name: float(items[f"train/{name}_loss"]) for name in _DETECT_LOSSES}
        return {"loss": sum(parts.values()), **{f"{name}_loss": v for name, v in parts.items()}}
    return {"loss": float(items["train/loss"])}


def _add_callbacks(model: Any, run: TrainContext, kind: str) -> None:
    beat = _Beat(run)

    def on_train_start(trainer: Any) -> None:
        beat()

    def on_train_epoch_start(trainer: Any) -> None:
        run.check_abort()
        run.heartbeat()

    def on_train_batch_end(trainer: Any) -> None:
        run.check_abort()
        beat()

    def on_fit_epoch_end(trainer: Any) -> None:
        run.check_abort()
        # Ultralytics fires this once more after the final validation on the best weights, with no validation
        # loss and the last epoch's number again. That is not a new epoch.
        validation = validation_metrics(kind, trainer.metrics or {})
        if validation is None:
            return
        epoch = int(trainer.epoch) + 1
        training = train_metrics(kind, trainer)
        if training is not None:
            run.report(epoch, "train", training)
        run.report(epoch, "validation", validation)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _class_names(model: Any, class_names: list[str], kind: str) -> list[str]:
    """The real class name per model index, as the trained model itself reports them.

    Classification: `model.names` holds the zero-padded class FOLDER names `data.py` wrote, in the index order
    the model learned; each is an index into the snapshot's class list. Detection: `data.yaml` carried the real
    names, so `model.names` already is the list.
    """
    names = [model.names[i] for i in sorted(model.names)]
    if len(names) != len(class_names):
        raise RuntimeError(f"The trained model has {len(names)} classes but the snapshot has {len(class_names)}")
    if kind == "detect":
        return names
    return [class_names[int(folder)] for folder in names]


def train(run: TrainContext) -> YoloLoadedModel:
    cfg = run.config
    workdir = Path(run.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        beat = _Beat(run)

        def progress() -> None:
            run.check_abort()
            beat()

        df = pd.read_parquet(run.dataset_uri)
        run.heartbeat()
        kind = cfg["task"]
        common = dict(
            image_column=cfg["imageColumn"],
            split_column=C.SPLIT_COLUMN_NAME,
            item_id_column=C.ITEM_ID_COLUMN_NAME,
            class_names=cfg["classNames"],
            progress=progress,
        )
        if kind == "detect":
            dataset = data.materialize_detection(df, str(workdir / "data"), boxes_column=cfg["labelColumn"], **common)
            # Detection trains from a data.yaml that points at the image and label trees.
            data_arg = dataset.data_yaml
        else:
            dataset = data.materialize_classification(
                df, str(workdir / "data"), label_column=cfg["labelColumn"], **common
            )
            data_arg = dataset.root
        run.check_abort()

        model = _yolo_class()(runtime.weights_ref(cfg["weights"]))
        _add_callbacks(model, run, kind)
        runs_dir = workdir / "runs"
        with tracer.start_as_current_span("yolo.train"):
            model.train(
                **cfg["train"],
                data=data_arg,
                project=str(runs_dir),
                name="run",
                exist_ok=True,
                plots=False,
                verbose=False,
                workers=runtime.dataloader_workers(),
                device=runtime.device(),
            )

        run_dir = runs_dir / "run"
        weights = run_dir / "weights" / runtime.WEIGHTS_FILENAME
        if not weights.is_file():
            weights = run_dir / "weights" / "last.pt"

        meta = {
            "schemaVersion": 1,
            "task": cfg["task"],
            "imgsz": cfg["train"]["imgsz"],
            "imageColumn": cfg["imageColumn"],
            "outputName": cfg["labelColumn"],
            "names": _class_names(model, cfg["classNames"], kind),
        }
        stage = workdir / "upload"
        stage.mkdir()
        shutil.copyfile(weights, stage / runtime.WEIGHTS_FILENAME)
        (stage / runtime.METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        for name in _UPLOADED:
            if (run_dir / name).is_file():
                shutil.copyfile(run_dir / name, stage / name)
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
