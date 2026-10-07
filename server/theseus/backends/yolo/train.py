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

from theseus import constants as C
from theseus.backends.base import TrainContext
from theseus.backends.yolo import data, runtime
from theseus.backends.yolo.model import YoloLoadedModel
from theseus.backends.yolo.model import load as load_model
from theseus.services import storage

tracer = trace.get_tracer("theseus")

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


def _add_callbacks(model: Any, run: TrainContext) -> None:
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
        metrics = trainer.metrics or {}
        # Ultralytics fires this once more after the final validation on the best weights, with no
        # `val/loss` and the last epoch's number again. That is not a new epoch.
        if "val/loss" not in metrics:
            return
        epoch = int(trainer.epoch) + 1
        validation = {"loss": metrics["val/loss"]}
        for source, name in (("metrics/accuracy_top1", "accuracy"), ("metrics/accuracy_top5", "top5Accuracy")):
            if source in metrics:
                validation[name] = metrics[source]
        if trainer.tloss is not None:
            run.report(epoch, "train", {"loss": float(trainer.label_loss_items(trainer.tloss)["train/loss"])})
        run.report(epoch, "validation", {k: float(v) for k, v in validation.items()})

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _class_names(model: Any, class_names: list[str]) -> list[str]:
    """The real class name per model index. `model.names` holds the zero-padded class FOLDER names `data.py`
    wrote, in the index order the model learned; each is an index into the snapshot's class list."""
    folder_names = [model.names[i] for i in sorted(model.names)]
    if len(folder_names) != len(class_names):
        raise RuntimeError(f"The trained model has {len(folder_names)} classes but the snapshot has {len(class_names)}")
    return [class_names[int(folder)] for folder in folder_names]


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
        dataset = data.materialize_classification(
            df,
            str(workdir / "data"),
            image_column=cfg["imageColumn"],
            label_column=cfg["labelColumn"],
            split_column=C.SPLIT_COLUMN_NAME,
            item_id_column=C.ITEM_ID_COLUMN_NAME,
            class_names=cfg["classNames"],
            progress=progress,
        )
        run.check_abort()

        model = _yolo_class()(runtime.weights_ref(cfg["weights"]))
        _add_callbacks(model, run)
        runs_dir = workdir / "runs"
        with tracer.start_as_current_span("yolo.train"):
            model.train(
                **cfg["train"],
                data=dataset.root,
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
            "names": _class_names(model, cfg["classNames"]),
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
