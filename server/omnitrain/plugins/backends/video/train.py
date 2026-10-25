"""The video half of training: download the clips, fine-tune a torchvision classifier, report progress, upload.

One epoch shows each training clip once, as ONE random clip of `numFrames` frames from a random place in the video
(with a random crop, and a random flip if enabled), so a long video is seen from a different part each epoch.
Validation uses a single centred clip per video to stay fast; the final evaluation uses `clipsPerVideo` clips.

Progress, heartbeats and the abort check run in the training loop itself, so `TrainingAborted` from
`run.check_abort()` stops training the same way it does for the other backends.
"""

import contextlib
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

from omnitrain.core import constants as C
from omnitrain.plugins.backends.base import TrainContext
from omnitrain.plugins.backends.video import custom, data, frames, hf, network, runtime
from omnitrain.plugins.backends.video.model import VideoLoadedModel, clip_spec_from_meta
from omnitrain.plugins.backends.video.model import load as load_model
from omnitrain.plugins.backends.video.schedule import lr_factor
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


class _Stepper:
    """Everything one optimizer step needs beyond the optimizer: the learning-rate curve, gradient clipping and mixed
    precision (a loss scaler and `autocast`, both only on a CUDA device, where half precision is worth it)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        device: str,
        hp: dict[str, Any],
        steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self._parameters = parameters
        self._clip = hp.get("gradClipNorm")
        self.amp = bool(hp.get("mixedPrecision")) and device.startswith("cuda")
        self._device_type = "cuda" if device.startswith("cuda") else "cpu"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        total = hp["epochs"] * steps_per_epoch
        warmup = round(float(hp.get("warmupEpochs") or 0) * steps_per_epoch)
        schedule = hp.get("lrSchedule") or "constant"
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total, warmup, schedule)
        )

    def autocast(self) -> Any:
        return torch.autocast(self._device_type, dtype=torch.float16, enabled=self.amp)

    def step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(loss).backward()
        if self._clip is not None:
            # Gradients are still multiplied by the loss scale; undo that first so the norm is the real one.
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._parameters, self._clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()
        self._scheduler.step()


def _run_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    device: str,
    run: TrainContext,
    beat: _Beat,
    stepper: _Stepper | None,
) -> tuple[float, float]:
    """One pass over `loader`: trains when a stepper is given, otherwise only scores. Returns (loss, accuracy)."""
    training = stepper is not None
    net.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    for clips, labels in loader:
        run.check_abort()
        beat()
        clips, labels = clips.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            with stepper.autocast() if stepper is not None else contextlib.nullcontext():
                logits = net(clips)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        if stepper is not None:
            stepper.step(loss)
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

        # A bring-your-own model dictates its own frames, image size and normalisation (see custom.py).
        bring_your_own = cfg.get("custom")
        clip: dict[str, Any] = dict(cfg["clip"])
        if bring_your_own:
            clip.update(custom.clip_settings(bring_your_own["path"]))
        meta: dict[str, Any] = {
            "schemaVersion": 1,
            "task": cfg["task"],
            "model": cfg["model"],
            # True: the trained network is a Hugging Face folder (config.json + model.safetensors), not model.pt.
            "hf": bool(bring_your_own),
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

        if bring_your_own:
            net = hf.build(bring_your_own["path"], class_names)
            if hp["freezeBackbone"]:
                hf.freeze_backbone(net)
        else:
            net = network.build_model(
                cfg["model"], len(class_names), pretrained=cfg["pretrained"], freeze_backbone=hp["freezeBackbone"]
            )
        net = net.to(device)
        trainable = [p for p in net.parameters() if p.requires_grad]
        # Each optimizer keeps its own weight decay unless one was asked for: SGD 1e-4, and AdamW torch's 0.01.
        decay = hp.get("weightDecay")
        if hp["optimizer"] == "SGD":
            optimizer: torch.optim.Optimizer = torch.optim.SGD(
                trainable, lr=hp["learningRate"], momentum=0.9, weight_decay=1e-4 if decay is None else decay
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable, lr=hp["learningRate"], **({} if decay is None else {"weight_decay": decay})
            )
        stepper = _Stepper(optimizer, trainable, device, hp, len(train_loader))

        best_loss, best_state, stale = float("inf"), None, 0
        with tracer.start_as_current_span("video.train"):
            for epoch in range(1, hp["epochs"] + 1):
                run.check_abort()
                run.heartbeat()
                loss, accuracy = _run_epoch(net, train_loader, device, run, beat, stepper)
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
        if bring_your_own:
            hf.save(net, str(stage))
        else:
            torch.save(net.state_dict(), stage / runtime.WEIGHTS_FILENAME)
        (stage / runtime.METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
