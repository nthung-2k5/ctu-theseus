"""The video half of training: download the clips, fine-tune a torchvision classifier, report progress, upload.

One epoch shows each training clip once, as ONE random clip of `numFrames` frames from a random place in the video
(with a random crop, and a random flip if enabled), so a long video is seen from a different part each epoch.
Validation uses a single centred clip per video to stay fast; the final evaluation uses `clipsPerVideo` clips.

Progress, heartbeats and the abort check run in the training loop itself, so `TrainingAborted` from
`run.check_abort()` stops training the same way it does for the other backends.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from opentelemetry import trace
from torch.utils.data import DataLoader, Dataset

from omnitrain import constants as C
from omnitrain.backends.base import TrainContext
from omnitrain.backends.video import data, frames, network, runtime
from omnitrain.backends.video.model import VideoLoadedModel, clip_spec_from_meta
from omnitrain.backends.video.model import load as load_model
from omnitrain.services import storage

tracer = trace.get_tracer("omnitrain")
logger = logging.getLogger(__name__)

# The run is presumed hung when it goes `run_heartbeat_timeout_seconds` (15 min) without a heartbeat, which a long
# epoch can exceed. Beat from the batch loop too, this often.
HEARTBEAT_EVERY_SECONDS = 10.0


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


class ClipDataset(Dataset):
    """One item per video. In training mode each access draws a fresh random clip; otherwise it is the centred clip."""

    def __init__(self, samples: list[data.Sample], spec: frames.ClipSpec, *, training: bool, flip: bool, seed: int):
        self._samples = samples
        self._spec = spec
        self._training = training
        self._flip = flip
        self._seed = seed
        self._draws = 0

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[i]
        rng = None
        flip = False
        if self._training:
            # A fresh generator per access, keyed by seed, item and how often it has been drawn: every epoch shows a
            # different part of the video, and a run with the same seed repeats exactly.
            self._draws += 1
            rng = np.random.default_rng([self._seed, i, self._draws])
            flip = self._flip and bool(rng.integers(0, 2))
        [clip] = frames.load_clips(sample.path, self._spec, 1, rng=rng, flip=flip)
        return frames.to_tensor(clip, self._spec), sample.class_index


def _run_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    device: str,
    run: TrainContext,
    beat: _Beat,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """One pass over `loader`: trains when an optimizer is given, otherwise only scores. Returns (loss, accuracy)."""
    training = optimizer is not None
    net.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    for clips, labels in loader:
        run.check_abort()
        beat()
        clips, labels = clips.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = net(clips)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += len(labels)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(run: TrainContext) -> VideoLoadedModel:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
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
        class_names: list[str] = cfg["classNames"]
        dataset = data.materialize(
            df,
            str(workdir / "data"),
            clip_column=cfg["clipColumn"],
            label_column=cfg["labelColumn"],
            split_column=C.SPLIT_COLUMN_NAME,
            item_id_column=C.ITEM_ID_COLUMN_NAME,
            class_names=class_names,
            progress=progress,
        )
        run.check_abort()

        clip: dict[str, Any] = cfg["clip"]
        meta: dict[str, Any] = {
            "schemaVersion": 1,
            "task": cfg["task"],
            "model": cfg["model"],
            "numFrames": clip["numFrames"],
            "frameStride": clip["frameStride"],
            "frameSize": clip["frameSize"],
            "mean": clip["mean"],
            "std": clip["std"],
            "clipsPerVideo": clip["clipsPerVideo"],
            "clipColumn": cfg["clipColumn"],
            "outputName": cfg["labelColumn"],
            # The model's class order: index i is class_names[i], because that is how the labels were encoded above.
            "names": class_names,
        }
        spec = clip_spec_from_meta(meta)
        seed = hp["seed"] if hp.get("seed") is not None else 0
        torch.manual_seed(seed)

        runtime.configure()
        device = runtime.device()
        batch_size = int(hp["batchSize"])
        workers = runtime.dataloader_workers()
        train_set = ClipDataset(dataset.samples["train"], spec, training=True, flip=hp["horizontalFlip"], seed=seed)
        val_set = ClipDataset(dataset.samples["val"], spec, training=False, flip=False, seed=seed)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            # A last batch of one clip breaks batch norm in training, and costs the epoch one clip at most.
            drop_last=len(train_set) > batch_size and len(train_set) % batch_size == 1,
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

        net = network.build_model(
            cfg["model"], len(class_names), pretrained=cfg["pretrained"], freeze_backbone=hp["freezeBackbone"]
        ).to(device)
        trainable = [p for p in net.parameters() if p.requires_grad]
        if hp["optimizer"] == "SGD":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                trainable, lr=hp["learningRate"], momentum=0.9, weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.AdamW(trainable, lr=hp["learningRate"])

        best_loss, best_state, stale = float("inf"), None, 0
        with tracer.start_as_current_span("video.train"):
            for epoch in range(1, hp["epochs"] + 1):
                run.check_abort()
                run.heartbeat()
                loss, accuracy = _run_epoch(net, train_loader, device, run, beat, optimizer)
                run.report(epoch, "train", {"loss": loss, "accuracy": accuracy})
                val_loss, val_accuracy = _run_epoch(net, val_loader, device, run, beat, None)
                run.report(epoch, "validation", {"loss": val_loss, "accuracy": val_accuracy})
                if val_loss < best_loss - 1e-6:
                    best_loss, stale = val_loss, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= hp["patience"]:
                        break

        stage = workdir / "upload"
        stage.mkdir()
        if best_state is not None:
            net.load_state_dict(best_state)
        torch.save(net.state_dict(), stage / runtime.WEIGHTS_FILENAME)
        (stage / runtime.METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
