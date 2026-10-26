"""The PyTorch half of training: prepare the data, fit a network, report progress, upload.

Four flows share `loop.fit` and differ in what a row is:

  * images: download them, fine-tune a torchvision classifier (or a bring-your-own Hugging Face one, which also dictates
    the image size and normalisation). One epoch shows each training image once, as a fresh random crop (and a random
    flip if enabled). Validation and the final evaluation use `images.preprocess`, the same resize-and-crop an export
    client applies.
  * tables: read the parquet, fit the per-column statistics on the TRAIN rows only, train a small perceptron that
    standardises inside the graph.
  * sounds: download and decode each recording once, train on random windows of its log-mel spectrogram.
  * texts: read the parquet, build the vocabulary from the TRAIN texts only, encode every text as word ids, train a
    word-level network from scratch. Nothing is downloaded: the text is in the snapshot. A bring-your-own transformer
    is fine-tuned on ids from its own tokenizer instead.

Progress, heartbeats and the abort check run in the training loop itself, so `TrainingAborted` from
`run.check_abort()` stops training the same way it does for the other backends.
"""

import json
import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from omnitrain.core import constants as C
from omnitrain.plugins.backends.base import ConfigError, TrainContext
from omnitrain.plugins.backends.pytorch import audio, custom, data, hf, images, loop, network, runtime, tabular, text
from omnitrain.plugins.backends.pytorch.model import PytorchLoadedModel
from omnitrain.plugins.backends.pytorch.model import load as load_model
from omnitrain.plugins.backends.pytorch.spec import AudioSpec, HFTextSpec, ImageSpec, TextSpec
from omnitrain.services import storage

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """One item per image. In training mode each access draws a fresh random crop; otherwise it is the plain
    resize-and-crop. The random draw is keyed by seed, epoch and item (never by a counter that a dataloader worker
    would copy), so every epoch differs, and a run with the same seed repeats exactly."""

    def __init__(self, samples: list[data.Sample], spec: ImageSpec, *, training: bool, flip: bool, seed: int):
        self._samples = samples
        self._spec = spec
        self._training = training
        self._flip = flip
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        if self._training:
            rng = np.random.default_rng([self._seed, self._epoch, index])
            array = images.augment(sample.path, self._spec, rng, flip=self._flip)
        else:
            array = images.preprocess(sample.path, self._spec)
        return torch.from_numpy(array), sample.class_index


class AudioDataset(Dataset):
    """One item per recording, read from the waveform cached at the model's rate. In training mode each access takes a
    random window and may mask parts of its spectrogram; otherwise it is the centred window. Keyed by seed, epoch and
    item, like `ImageDataset`."""

    def __init__(self, samples: list[data.Sample], spec: AudioSpec, *, training: bool, augment: bool, seed: int):
        self._samples = samples
        self._spec = spec
        self._training = training
        self._augment = augment
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        wave = np.load(sample.path)
        length = self._spec.length
        if self._training:
            rng = np.random.default_rng([self._seed, self._epoch, index])
            start = audio.random_start(len(wave), length, rng)
        else:
            start = audio.window_starts(len(wave), length, 1)[0]
        array = audio.spectrogram(audio.cut(wave, start, length), self._spec)
        if self._training and self._augment:
            array = audio.spec_augment(array, rng)
        return torch.from_numpy(array)[None], sample.class_index


# Recordings the spectrogram's mean and std are measured on: enough to be steady, few enough to be quick.
NORMALIZATION_SAMPLE = 500


def fit_normalization(samples: list[data.Sample], spec: AudioSpec, seed: int) -> AudioSpec:
    """`spec` with the scalar mean and std of the TRAIN recordings' log-mel values (their centred windows), so nothing
    about validation or test leaks into the scaling."""
    order = np.random.default_rng(seed).permutation(len(samples))[:NORMALIZATION_SAMPLE]
    total, squares, count = 0.0, 0.0, 0
    for i in order:
        wave = np.load(samples[int(i)].path)
        values = audio.log_mel(audio.cut(wave, audio.window_starts(len(wave), spec.length, 1)[0], spec.length), spec)
        total += float(values.sum(dtype=np.float64))
        squares += float(np.square(values, dtype=np.float64).sum())
        count += values.size
    mean = total / max(count, 1)
    return spec.with_normalization(mean, math.sqrt(max(squares / max(count, 1) - mean * mean, 0.0)))


@dataclass
class Setup:
    """What one flow hands the shared part: the network and its data, what is optimised, and what to save."""

    net: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    objective: loop.Objective
    # The `names.json` content: the input spec, the model's own class order and the network's shape.
    meta: dict[str, Any]
    on_epoch: Callable[[int], None] | None = None


def _image_setup(
    run: TrainContext, df: pd.DataFrame, workdir: Path, beat: loop.Beat, seed: int, class_names: list[str]
) -> Setup:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]

    def progress() -> None:
        run.check_abort()
        beat()

    dataset = data.materialize(
        df,
        str(workdir / "data"),
        file_column=cfg["imageColumn"],
        label_column=cfg["labelColumn"],
        split_column=C.SPLIT_COLUMN_NAME,
        item_id_column=C.ITEM_ID_COLUMN_NAME,
        class_names=class_names,
        progress=progress,
    )
    run.check_abort()

    spec = ImageSpec.from_meta(cfg["image"])
    bring_your_own = cfg.get("custom")
    if bring_your_own:
        # A bring-your-own model dictates its own image size and normalisation (see custom.py).
        found = custom.image_settings(bring_your_own["path"])
        spec = ImageSpec(size=found["imageSize"], mean=tuple(found["mean"]), std=tuple(found["std"]))
    meta = {
        "schemaVersion": 1,
        "task": cfg["task"],
        "model": cfg["model"],
        # True: the trained network is a Hugging Face folder (config.json + model.safetensors), not model.pt.
        "hf": bool(bring_your_own),
        **spec.to_meta(),
        "imageColumn": cfg["imageColumn"],
        "outputName": cfg["labelColumn"],
        # The model's class order: index i is class_names[i], because that is how the labels were encoded above.
        "names": class_names,
    }
    runtime.configure()
    batch_size = int(hp["batchSize"])
    workers = runtime.dataloader_workers()
    train_set = ImageDataset(dataset.samples["train"], spec, training=True, flip=hp["horizontalFlip"], seed=seed)
    val_set = ImageDataset(dataset.samples["val"], spec, training=False, flip=False, seed=seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        # A last batch of one image breaks batch norm in training, and costs the epoch one image at most.
        drop_last=len(train_set) > batch_size and len(train_set) % batch_size == 1,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    net: torch.nn.Module
    if bring_your_own:
        net = hf.build(bring_your_own["path"], class_names)
        if hp["freezeBackbone"]:
            hf.freeze_backbone(net)
    else:
        net = network.build_model(
            cfg["model"], len(class_names), pretrained=cfg["pretrained"], freeze_backbone=hp["freezeBackbone"]
        )
    return Setup(net, train_loader, val_loader, loop.Classification(), meta, on_epoch=train_set.set_epoch)


def _audio_setup(
    run: TrainContext, df: pd.DataFrame, workdir: Path, beat: loop.Beat, seed: int, class_names: list[str]
) -> Setup:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    initial = AudioSpec.from_meta(cfg["audio"])

    def progress() -> None:
        run.check_abort()
        beat()

    def prepare(local: str) -> str:
        """Decode, mix down and resample once, here in the download thread, and keep the waveform: no epoch repeats
        that work, and an undecodable or empty file becomes a skipped row instead of a failed run."""
        wave = audio.load_waveform(local, initial)
        if len(wave) == 0:
            raise ValueError("the recording has no samples")
        cached = local + ".npy"
        np.save(cached, wave)
        os.remove(local)
        return cached

    dataset = data.materialize(
        df,
        str(workdir / "data"),
        file_column=cfg["audioColumn"],
        label_column=cfg["labelColumn"],
        split_column=C.SPLIT_COLUMN_NAME,
        item_id_column=C.ITEM_ID_COLUMN_NAME,
        class_names=class_names,
        progress=progress,
        prepare=prepare,
        noun="recordings",
        default_ext=".wav",
    )
    run.check_abort()

    spec = fit_normalization(dataset.samples["train"], initial, seed)
    meta = {
        "schemaVersion": 1,
        "task": cfg["task"],
        "model": cfg["model"],
        **spec.to_meta(),
        "audioColumn": cfg["audioColumn"],
        "outputName": cfg["labelColumn"],
        # The model's class order: index i is class_names[i], because that is how the labels were encoded above.
        "names": class_names,
    }
    runtime.configure()
    batch_size = int(hp["batchSize"])
    workers = runtime.dataloader_workers()
    train_set = AudioDataset(dataset.samples["train"], spec, training=True, augment=hp["specAugment"], seed=seed)
    val_set = AudioDataset(dataset.samples["val"], spec, training=False, augment=False, seed=seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        # A last batch of one recording breaks batch norm in training, and costs the epoch one at most.
        drop_last=len(train_set) > batch_size and len(train_set) % batch_size == 1,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    net = network.build_audio(
        cfg["model"], len(class_names), pretrained=cfg["pretrained"], freeze_backbone=hp["freezeBackbone"]
    )
    return Setup(net, train_loader, val_loader, loop.Classification(), meta, on_epoch=train_set.set_epoch)


def _tabular_setup(run: TrainContext, df: pd.DataFrame, class_names: list[str]) -> Setup:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    kind: str = cfg["task"]
    regress = kind == "regress"
    columns: list[str] = cfg["inputColumns"]

    splits = tabular.prepare(
        df,
        kind=kind,
        columns=columns,
        label_column=cfg["labelColumn"],
        class_names=class_names,
        split_column=C.SPLIT_COLUMN_NAME,
    )
    for name in ("train", "validation"):
        if len(splits[name]) == 0:
            raise ConfigError(f"The {name} split has no labelled rows, which training needs")
    if not regress and len(set(splits["train"].y.tolist())) < 2:
        raise ConfigError("The train split has only one class, so there is nothing to tell apart")

    # Fitted on the train rows alone, so nothing about validation or test leaks into the scaling.
    spec = tabular.fit_spec(columns, splits["train"].x, splits["train"].y if regress else None)
    meta = {
        "schemaVersion": 1,
        "task": kind,
        "model": cfg["model"],
        **spec.to_meta(),
        "network": cfg["network"],
        "outputName": cfg["labelColumn"],
        "names": [] if regress else class_names,
    }
    net = network.build_tabular(
        spec, list(cfg["network"]["hidden"]), float(cfg["network"]["dropout"]), len(class_names), regress=regress
    )

    objective: loop.Objective
    if regress:
        objective = loop.Regression(spec.target_std)
    else:
        weights = None
        if hp.get("useClassWeights"):
            weights = torch.from_numpy(tabular.balanced_class_weights(splits["train"].y, len(class_names)))
        objective = loop.Classification(weights)

    batch_size = int(hp["batchSize"])

    def loader(split: str, *, shuffle: bool) -> DataLoader:
        part = splits[split]
        return DataLoader(
            TensorDataset(torch.from_numpy(part.x), torch.from_numpy(part.y)), batch_size=batch_size, shuffle=shuffle
        )

    return Setup(net, loader("train", shuffle=True), loader("validation", shuffle=False), objective, meta)


def _text_splits(run: TrainContext, df: pd.DataFrame, class_names: list[str]) -> dict[str, text.Split]:
    cfg = run.config
    splits = text.prepare(
        df,
        text_column=cfg["textColumn"],
        label_column=cfg["labelColumn"],
        class_names=class_names,
        split_column=C.SPLIT_COLUMN_NAME,
    )
    for name in ("train", "validation"):
        if len(splits[name]) == 0:
            raise ConfigError(
                f"The {name} split has no usable texts, which training needs. Check the snapshot's split settings "
                "and that its items are labelled."
            )
    if len(set(splits["train"].y.tolist())) < 2:
        raise ConfigError("The train split has only one class, so there is nothing to tell apart")
    return splits


def _hf_text_setup(run: TrainContext, df: pd.DataFrame, class_names: list[str]) -> Setup:
    """A bring-your-own transformer: its own tokenizer turns each text into ids once, and it is fine-tuned on them."""
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    path: str = cfg["custom"]["path"]
    splits = _text_splits(run, df, class_names)

    max_tokens = int(cfg["text"]["maxTokens"])
    limit = custom.text_token_limit(path)
    if max_tokens > limit:
        raise ConfigError(
            f"This model reads at most {limit} tokens of a text, but the run asks for {max_tokens}. Lower the number "
            "of words read per text."
        )
    tokenizer_json, pad_id, pad_token = hf.read_tokenizer(path)
    spec = HFTextSpec(max_tokens=max_tokens, pad_id=pad_id, pad_token=pad_token, tokenizer=tokenizer_json)
    meta = {
        "schemaVersion": 1,
        "task": cfg["task"],
        "model": cfg["model"],
        # True: the trained network is a Hugging Face folder (config.json + model.safetensors), not model.pt.
        "hf": True,
        **spec.to_meta(),
        "textColumn": cfg["textColumn"],
        "outputName": cfg["labelColumn"],
        # The model's class order: index i is class_names[i], because that is how the labels were encoded above.
        "names": class_names,
    }
    net = hf.build_text(path, class_names, pad_id)
    if hp["freezeBackbone"]:
        hf.freeze_backbone(net)

    weights = None
    if hp.get("useClassWeights"):
        weights = torch.from_numpy(tabular.balanced_class_weights(splits["train"].y, len(class_names)))
    tokenizer = hf.make_tokenizer(spec)
    batch_size = int(hp["batchSize"])

    def loader(split: str, *, shuffle: bool) -> DataLoader:
        part = splits[split]
        return DataLoader(
            TensorDataset(hf.encode_texts(tokenizer, part.texts), torch.from_numpy(part.y)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return Setup(
        net, loader("train", shuffle=True), loader("validation", shuffle=False), loop.Classification(weights), meta
    )


def _text_setup(run: TrainContext, df: pd.DataFrame, class_names: list[str]) -> Setup:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    settings: dict[str, Any] = cfg["text"]
    splits = _text_splits(run, df, class_names)

    # Built from the train texts alone, so nothing about validation or test leaks into what the model knows.
    vocabulary = text.fit_vocabulary(
        splits["train"].texts, max_words=int(settings["vocabSize"]), min_count=int(settings["minWordCount"])
    )
    spec = TextSpec(max_tokens=int(settings["maxTokens"])).with_vocabulary(vocabulary)
    dropout = float(cfg["network"]["dropout"])
    meta = {
        "schemaVersion": 1,
        "task": cfg["task"],
        "model": cfg["model"],
        **spec.to_meta(),
        "textColumn": cfg["textColumn"],
        "network": {"dropout": dropout},
        "outputName": cfg["labelColumn"],
        # The model's class order: index i is class_names[i], because that is how the labels were encoded above.
        "names": class_names,
    }
    net = network.build_text(cfg["model"], spec.size, len(class_names), dropout)

    weights = None
    if hp.get("useClassWeights"):
        weights = torch.from_numpy(tabular.balanced_class_weights(splits["train"].y, len(class_names)))
    batch_size = int(hp["batchSize"])

    def loader(split: str, *, shuffle: bool) -> DataLoader:
        part = splits[split]
        return DataLoader(
            TensorDataset(torch.from_numpy(text.encode(part.texts, spec)), torch.from_numpy(part.y)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return Setup(
        net, loader("train", shuffle=True), loader("validation", shuffle=False), loop.Classification(weights), meta
    )


def train(run: TrainContext) -> PytorchLoadedModel:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    workdir = Path(run.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        beat = loop.Beat(run)
        df = pd.read_parquet(run.dataset_uri)
        run.heartbeat()
        class_names: list[str] = cfg["classNames"]
        seed = hp["seed"] if hp.get("seed") is not None else 0
        torch.manual_seed(seed)

        if cfg["inputKind"] == "tabular":
            setup = _tabular_setup(run, df, class_names)
        elif cfg["inputKind"] == "text":
            setup = (_hf_text_setup if cfg.get("custom") else _text_setup)(run, df, class_names)
        elif cfg["inputKind"] == "audio":
            setup = _audio_setup(run, df, workdir, beat, seed, class_names)
        else:
            setup = _image_setup(run, df, workdir, beat, seed, class_names)

        device = runtime.device()
        net = setup.net.to(device)
        loop.fit(
            run, net, setup.train_loader, setup.val_loader,
            hp=hp, device=device, beat=beat, objective=setup.objective, span="pytorch.train", on_epoch=setup.on_epoch,
        )  # fmt: skip

        stage = workdir / "upload"
        stage.mkdir()
        if setup.meta.get("hf"):
            hf.save(net, str(stage))
        else:
            torch.save(net.state_dict(), stage / runtime.WEIGHTS_FILENAME)
        (stage / runtime.METADATA_FILENAME).write_text(json.dumps(setup.meta, indent=2), encoding="utf-8")
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
