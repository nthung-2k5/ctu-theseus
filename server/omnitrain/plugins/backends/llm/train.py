"""The language-model half of training: read the snapshot, fine-tune LoRA adapters on a causal model, merge, upload.

An epoch shows every training example once. A run is judged by the mean cross entropy over the answer tokens of the
validation split (the prompt is not graded), the quantity `events/writer.py` already uses to find a run's best epoch;
training keeps the adapters of the epoch with the lowest one. When it ends they are folded into the weights and the
whole model is saved as a Hugging Face folder, so a trained run loads and serves from its results alone, wherever the
base model's files were (which is what lets it train on a remote worker).

Needs only S3 and the `TrainContext` callbacks, like every backend's `train()`.
"""

import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from omnitrain.core import constants as C
from omnitrain.plugins.backends.base import ConfigError, TrainContext
from omnitrain.plugins.backends.llm import custom, data, network, runtime
from omnitrain.plugins.backends.llm.model import LlmLoadedModel
from omnitrain.plugins.backends.llm.model import load as load_model
from omnitrain.plugins.backends.pytorch import loop
from omnitrain.services import storage

logger = logging.getLogger(__name__)


def _splits(run: TrainContext, df: pd.DataFrame) -> dict[str, data.Rows]:
    cfg = run.config
    splits = data.prepare(df, cfg["inputColumns"], cfg["outputColumn"], C.SPLIT_COLUMN_NAME)
    for name in ("train", "validation"):
        if len(splits[name]) == 0:
            raise ConfigError(
                f"The {name} split has no usable rows, which training needs. Check the snapshot's split settings and "
                "that its items have an answer written."
            )
    return splits


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(run: TrainContext) -> LlmLoadedModel:
    cfg = run.config
    hp: dict[str, Any] = cfg["train"]
    text: dict[str, Any] = cfg["text"]
    lora: dict[str, Any] = cfg["lora"]
    path: str = cfg["custom"]["path"]
    quantization: str = cfg.get("quantization", "none")
    if reason := network.quantization_unavailable(quantization):
        raise ConfigError(f"The base model cannot be held in {quantization} here: {reason}")
    workdir = Path(run.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        beat = loop.Beat(run)
        df = pd.read_parquet(run.dataset_uri)
        run.heartbeat()

        limit = custom.position_limit(custom.check_files(Path(path)))
        if limit is not None and int(text["maxLength"]) > limit:
            raise ConfigError(
                f"This model reads at most {limit} tokens at once, but the run asks for {text['maxLength']}. Lower the "
                "longest prompt and answer."
            )
        splits = _splits(run, df)
        seed = hp["seed"] if hp.get("seed") is not None else 0
        torch.manual_seed(seed)

        tokenizer = network.load_tokenizer(path)
        pad_id = network.pad_id(tokenizer)
        examples = {
            name: [
                data.encode_example(
                    tokenizer, p, a, max_length=int(text["maxLength"]), max_new_tokens=int(text["maxNewTokens"])
                )
                for p, a in zip(rows.prompts, rows.answers, strict=True)
            ]
            for name, rows in splits.items()
            if name != "test"
        }
        run.check_abort()

        device = runtime.device()
        quantized = quantization != "none"
        base = network.load_base(path, runtime.dtype(), quantization)
        base = network.prepare(base, quantized=quantized, checkpointing=bool(hp.get("gradientCheckpointing")))
        peft_model = network.add_lora(base, int(lora["rank"]), int(lora["alpha"]), float(lora["dropout"]))
        # A quantised model is already on the GPU, in the right number format; moving it again is refused.
        net = network.CausalNet(peft_model)
        if not quantized:
            net = net.to(device)
        batch_size = int(hp["batchSize"])
        collate = data.collate(pad_id)
        train_loader = DataLoader(examples["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(examples["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate)
        loop.fit(
            run, net, train_loader, val_loader,
            hp=hp, device=device, beat=beat, objective=loop.Tagging(), span="llm.train", keep=network.is_adapter,
        )  # fmt: skip

        stage = workdir / "upload"
        if quantized:
            # Weights held in 4 or 8 bits cannot absorb the adapters: write them down, let go of the quantised model,
            # and merge them into a full-precision copy of the base.
            adapter = network.save_adapter(peft_model)
            del net, peft_model, base, train_loader, val_loader
            _free_memory()
            merged = network.merge_saved(path, adapter.name, runtime.dtype())
            adapter.cleanup()
        else:
            merged = network.merge(peft_model)
        network.save(merged, tokenizer, str(stage))
        meta = {
            "schemaVersion": 1,
            "task": cfg["task"],
            "model": cfg["model"],
            "inputKind": "llm",
            "inputColumns": cfg["inputColumns"],
            "outputColumn": cfg["outputColumn"],
            "text": text,
            "lora": lora,
        }
        (stage / runtime.METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # The training copy of the model is as large as the one about to be loaded for serving; free it first.
        del merged
        if not quantized:
            del net, peft_model, base, train_loader, val_loader
        _free_memory()
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
