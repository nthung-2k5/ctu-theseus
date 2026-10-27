"""Runs the language model in this bundle: text in, text out.

The model is a Hugging Face folder (`model/`: weights, tokenizer and config), so there is no graph to re-implement.
What this file repeats is the recipe the platform wrapped around it, which `preprocessing.json` describes under
`generation` (recipe `causal_lm_prompt_v1`): how the input fields become a prompt, how a long prompt is cut, and the
greedy decoding of the answer. `verify.py` checks that it writes what the platform wrote for the bundled sample.

    pip install torch transformers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECIPE = "causal_lm_prompt_v1"
_HERE = Path(__file__).parent


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def build_prompt(input_columns: list[str], values: list[Any], output_column: str) -> str:
    """The text the model continues: each input field under a `### name:` heading, then the answer's heading."""
    sections = [f"### {name}:\n{_clean(value)}\n" for name, value in zip(input_columns, values)]
    return "\n".join(sections) + f"\n### {output_column}:\n"


def clip_prompt(ids: list[int], room: int, bos_id: int | None) -> list[int]:
    """The last `room` ids of a prompt (its end holds the heading the answer continues), keeping a leading
    beginning-of-text id when there is one."""
    if len(ids) <= room:
        return ids
    if bos_id is not None and ids and ids[0] == bos_id and room > 1:
        return [bos_id, *ids[-(room - 1) :]]
    return ids[-room:]


class OmniTrainLLM:
    """Loads `model/` next to this file (or from `directory`, a bundle's root) and writes answers."""

    def __init__(self, directory: str | Path | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        root = Path(directory) if directory else _HERE
        manifest = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        generation = manifest.get("generation") or {}
        if generation.get("recipe") != RECIPE:
            raise NotImplementedError(
                f"This client implements the {RECIPE!r} recipe, but preprocessing.json asks for "
                f"{generation.get('recipe')!r}. Adapt the client to the recipe it describes."
            )
        self.input_columns: list[str] = list(generation["inputColumns"])
        self.output_column: str = generation["outputColumn"]
        self._room = int(generation["promptRoom"])
        self._max_new_tokens = int(generation["maxNewTokens"])

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # What the platform runs a model in: bfloat16 on a GPU that has it natively, else float32. A different number
        # format can change which token is the most likely, and so the answer.
        native = self._device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        dtype = torch.bfloat16 if native else torch.float32
        model_dir = root / "model"
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = (
            AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, local_files_only=True).to(self._device).eval()
        )
        tokenizer = self._tokenizer
        self._pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    def prompt(self, inputs: str | dict[str, Any]) -> str:
        """A text (when the model has one input field) or a `{field: text}` dict, as the prompt the model continues."""
        if isinstance(inputs, dict):
            values = [inputs.get(column) for column in self.input_columns]
        elif len(self.input_columns) == 1:
            values = [inputs]
        else:
            raise ValueError(f"This model takes {len(self.input_columns)} fields: pass a dict with {self.input_columns}")
        return build_prompt(self.input_columns, values, self.output_column)

    def predict(self, inputs: str | dict[str, Any]) -> str:
        """The model's answer: its most likely continuation, up to the end-of-text token or `maxNewTokens` tokens."""
        torch = self._torch
        tokenizer = self._tokenizer
        ids = list(tokenizer(self.prompt(inputs), add_special_tokens=True)["input_ids"])
        ids = clip_prompt(ids, self._room, tokenizer.bos_token_id)
        batch = torch.tensor([ids], device=self._device)
        with torch.no_grad():
            written = self._model.generate(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(written[0, batch.shape[1] :], skip_special_tokens=True).strip()
