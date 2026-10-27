"""Snapshot rows -> token ids a causal language model trains on and is asked from. Imports torch.

A training example is the prompt then the answer then the end-of-text token, with the loss on the answer only (the
prompt is labelled `IGNORE`): the model is graded on what it writes, not on repeating the question. Prompt and
answer are tokenised separately so the boundary is the same whether the answer is known (training) or being written
(serving).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

from omnitrain.plugins.backends.llm import prompt

IGNORE = -100


@dataclass
class Rows:
    """The usable rows of one split: a prompt and the answer written for it."""

    prompts: list[str]
    answers: list[str]

    def __len__(self) -> int:
        return len(self.prompts)


@dataclass
class Example:
    ids: list[int]
    labels: list[int]


def prepare(df: pd.DataFrame, input_columns: Sequence[str], output_column: str, split_column: str) -> dict[str, Rows]:
    """Rows by split name. A row with no answer teaches nothing, and one with every input empty has no question."""
    splits = {name: Rows([], []) for name in ("train", "validation", "test")}
    for _, row in df.iterrows():
        split = splits.get(str(row[split_column]))
        answer = prompt.format_answer(row[output_column])
        values = [row[c] for c in input_columns]
        if split is None or not answer or not any(prompt.format_answer(v) for v in values):
            continue
        split.prompts.append(prompt.format_prompt(input_columns, values, output_column))
        split.answers.append(answer)
    return splits


def clip_prompt(ids: list[int], cap: int, bos_id: int | None) -> list[int]:
    """The last `cap` ids of a prompt (its end holds the heading the answer continues), keeping a leading
    beginning-of-text token if there is one."""
    if len(ids) <= cap:
        return ids
    if bos_id is not None and ids and ids[0] == bos_id and cap > 1:
        return [bos_id, *ids[-(cap - 1) :]]
    return ids[-cap:]


prompt_room = prompt.prompt_room


def encode_prompt(tokenizer: Any, text: str, cap: int) -> list[int]:
    """A prompt as ids, special tokens included, cut to at most `cap`."""
    return clip_prompt(list(tokenizer(text, add_special_tokens=True)["input_ids"]), cap, tokenizer.bos_token_id)


def encode_example(tokenizer: Any, text: str, answer: str, *, max_length: int, max_new_tokens: int) -> Example:
    """Prompt + answer + end-of-text, at most `max_length` ids. The answer is cut to `max_new_tokens` (end-of-text
    included). The prompt is cut to the room a LONGEST answer leaves, whatever this answer's length: serving cuts a
    prompt to exactly that (it cannot know how long the answer will be), so a prompt must look the same in both."""
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])[: max_new_tokens - 1]
    answer_ids.append(int(tokenizer.eos_token_id))
    prompt_ids = encode_prompt(tokenizer, text, prompt_room(max_length, max_new_tokens))
    return Example(ids=prompt_ids + answer_ids, labels=[IGNORE] * len(prompt_ids) + answer_ids)


def collate(pad_id: int) -> Callable[[list[Example]], tuple[torch.Tensor, torch.Tensor]]:
    """A batch padded on the right to its longest example: `[batch, 2, length]` (ids, attention mask) and the targets
    `[batch, length - 1]` (the labels shifted left by one, as `network.CausalNet` scores the next token)."""

    def run(batch: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
        length = max(len(e.ids) for e in batch)
        packed = torch.full((len(batch), 2, length), 0, dtype=torch.long)
        packed[:, 0] = pad_id
        targets = torch.full((len(batch), length - 1), IGNORE, dtype=torch.long)
        for i, e in enumerate(batch):
            n = len(e.ids)
            packed[i, 0, :n] = torch.tensor(e.ids)
            packed[i, 1, :n] = 1
            targets[i, : n - 1] = torch.tensor(e.labels[1:])
        return packed, targets

    return run


def left_pad(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompts padded on the LEFT to the longest (generation continues from the last position, so the text must end
    there): ids and the attention mask, both `[batch, length]`."""
    length = max(len(r) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, length - len(r) :] = torch.tensor(list(r))
        mask[i, length - len(r) :] = 1
    return ids, mask
