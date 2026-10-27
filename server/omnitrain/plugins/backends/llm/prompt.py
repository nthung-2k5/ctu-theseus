"""What the model is shown, as text (pure, no torch).

One fixed layout for every task: each input field under a `### name:` heading, then the answer's heading, which the
model continues. It is built from the snapshot's own column names and stored in `names.json`, so training, evaluation
and serving cannot disagree about it. A base model needs no chat template this way: it learns the layout from the
examples.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd


def _clean(value: Any) -> str:
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def format_prompt(input_names: Sequence[str], values: Sequence[Any], output_name: str) -> str:
    """The text the model continues. `values[i]` is the text of field `input_names[i]`."""
    if len(input_names) != len(values):
        raise ValueError("a prompt needs one value for each input field")
    sections = [f"### {name}:\n{_clean(value)}\n" for name, value in zip(input_names, values, strict=True)]
    return "\n".join(sections) + f"\n### {output_name}:\n"


def prompt_room(max_length: int, max_new_tokens: int) -> int:
    """The most tokens a prompt may have: what is left of `max_length` after the longest answer. The one number training
    and serving both cut a prompt to."""
    return max_length - max_new_tokens


def format_answer(value: Any) -> str:
    """The answer as the model should learn to write it (surrounding whitespace dropped)."""
    return _clean(value)
