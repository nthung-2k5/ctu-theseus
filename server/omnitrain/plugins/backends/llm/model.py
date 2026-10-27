"""A trained language model, loaded: writes an answer for a prompt built from the task's input fields. Imports torch."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from omnitrain.domain.predict import InferenceOutput
from omnitrain.plugins.backends.base import LoadedModel, OutputSpec
from omnitrain.plugins.backends.llm import data, network, prompt, runtime, tasks


class LlmLoadedModel(LoadedModel):
    def __init__(self, model: torch.nn.Module, tokenizer: Any, meta: dict[str, Any]) -> None:
        self._device = runtime.device()
        self._model = model.to(self._device).eval()
        self._tokenizer = tokenizer
        self._pad_id = network.pad_id(tokenizer)
        self._meta = meta
        self.input_columns = list(meta["inputColumns"])
        self.output = OutputSpec(name=meta["outputColumn"], kind="text")

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def column(self) -> str:
        return f"{self.output.name}_predictions"

    def prompts(self, frame: pd.DataFrame) -> list[str]:
        return [
            prompt.format_prompt(self.input_columns, [row[c] for c in self.input_columns], self.output.name)
            for _, row in frame.iterrows()
        ]

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        """The model's answer to each prompt: the most likely continuation, up to the end-of-text token or the longest
        answer the run was trained for."""
        text = self._meta["text"]
        cap = data.prompt_room(int(text["maxLength"]), int(text["maxNewTokens"]))
        answers: list[str] = []
        for start in range(0, len(prompts), tasks.GENERATE_BATCH):
            encoded = [
                data.encode_prompt(self._tokenizer, p, cap) for p in prompts[start : start + tasks.GENERATE_BATCH]
            ]
            ids, mask = data.left_pad(encoded, self._pad_id)
            written = self._model.generate(  # type: ignore[operator]
                input_ids=ids.to(self._device),
                attention_mask=mask.to(self._device),
                max_new_tokens=int(text["maxNewTokens"]),
                do_sample=False,
                pad_token_id=self._pad_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            for row in written[:, ids.shape[1] :]:
                answers.append(self._tokenizer.decode(row, skip_special_tokens=True).strip())
        return answers

    @torch.no_grad()
    def answer_loss(self, frame: pd.DataFrame, batch_size: int = tasks.GENERATE_BATCH) -> tuple[float, int]:
        """(summed cross entropy, number of answer tokens) with the written answers fed in (teacher forcing). The same
        quantity as the run's validation `loss`, so the two can be compared."""
        text = self._meta["text"]
        prompts = self.prompts(frame)
        answers = [prompt.format_answer(a) for a in frame[self.output.name]]
        examples = [
            data.encode_example(
                self._tokenizer, p, a, max_length=int(text["maxLength"]), max_new_tokens=int(text["maxNewTokens"])
            )
            for p, a in zip(prompts, answers, strict=True)
        ]
        collate = data.collate(self._pad_id)
        total, count = 0.0, 0
        for start in range(0, len(examples), batch_size):
            packed, targets = collate(examples[start : start + batch_size])
            logits = network.CausalNet(self._model)(packed.to(self._device))
            targets = targets.to(self._device)
            total += float(
                torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                    ignore_index=data.IGNORE, reduction="sum",
                ).item()
            )  # fmt: skip
            count += int((targets != data.IGNORE).sum().item())
        return total, count

    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._model.state_dict().values())

    def save(self, dest: str) -> None:
        """The model as the folder `load` reads: weights, tokenizer and `names.json`."""
        network.save(self._model, self._tokenizer, dest)
        (Path(dest) / runtime.METADATA_FILENAME).write_text(json.dumps(self._meta, indent=2), encoding="utf-8")

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({self.column: self.generate(self.prompts(frame))})

    def to_output(
        self, predictions: pd.DataFrame, *, top_k: int = 100, input_tokens: list[str] | None = None
    ) -> InferenceOutput:
        text = str(predictions[self.column].iloc[0]) if self.column in predictions.columns else ""
        return {"kind": "text", "feature": self.output.name, "text": text}

    def golden_prediction(self, predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, float]:
        # A written answer has no per-class confidences to compare; this backend produces no export bundle anyway.
        return {}

    def close(self) -> None:
        del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load(model_dir: str) -> LlmLoadedModel:
    root = Path(model_dir)
    meta = json.loads((root / runtime.METADATA_FILENAME).read_text(encoding="utf-8"))
    # Every tensor must be in the files: a trained model has nothing left to make up.
    return LlmLoadedModel(network.load_base(str(root), runtime.dtype()), network.load_tokenizer(str(root)), meta)
