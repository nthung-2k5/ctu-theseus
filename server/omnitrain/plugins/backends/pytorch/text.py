"""Text -> the token ids a PyTorch text classifier takes: the one definition train, serve and evaluate share, and the
recipe the export manifest spells out for a client (`manifest.py`).

The tokenizer is deliberately primitive so that a client in any language can repeat it exactly, with no Unicode tables
and no library: only the ASCII letters A-Z are lower-cased, and a token is a maximal run of characters that are not
separators. A separator is an ASCII control, space or punctuation character (everything below U+0080 that is not a
letter or a digit) or one of the few Unicode spaces in `EXTRA_SEPARATORS`. Every other character, which means every
letter outside ASCII (Vietnamese, accented Latin, Cyrillic, CJK, ...), is part of a token, so a Vietnamese word stays
whole. A language that does not put spaces between words (Chinese, Japanese) therefore gives one long token per
run of text; this model is not meant for those. The same rule is the same on UTF-16 code units, which is why a
client may work on either.

A text becomes `max_tokens` ids: the first `max_tokens` tokens, each its id in the vocabulary (`UNKNOWN` for a word the
model never saw), padded at the end with `PAD`. The vocabulary is built from the TRAIN texts only.

numpy and pandas only: no torch.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from omnitrain.plugins.backends.pytorch.spec import TextSpec

PAD = 0
UNKNOWN = 1
# Ids from here on are the vocabulary's words, in `TextSpec.vocab` order.
FIRST_WORD = 2

# Code points (inclusive ranges) that separate tokens: ASCII controls, space and punctuation (everything below
# U+0080 except the digits 0-9, A-Z and a-z), then the Unicode spaces: no-break space, the en/em/thin/... spaces
# U+2000..U+200A, the line and paragraph separators, the narrow no-break space, the medium mathematical space and the
# ideographic space. The export manifest hands this exact list to a client.
SEPARATOR_RANGES: tuple[tuple[int, int], ...] = (
    (0x00, 0x2F), (0x3A, 0x40), (0x5B, 0x60), (0x7B, 0x7F),
    (0xA0, 0xA0), (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
)  # fmt: skip
_SEPARATORS = re.compile(
    "[" + "".join(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}" for lo, hi in SEPARATOR_RANGES) + "]+"
)
_LOWER = {c: c + 32 for c in range(ord("A"), ord("Z") + 1)}


def tokenize(text: str) -> list[str]:
    return [t for t in _SEPARATORS.split(text.translate(_LOWER)) if t]


def fit_vocabulary(texts: Iterable[str], *, max_words: int, min_count: int) -> list[str]:
    """The `max_words` most frequent tokens (counting every occurrence) that occur at least `min_count` times, most
    frequent first and alphabetical among equals, so the same texts always give the same vocabulary."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    kept = [(word, n) for word, n in counts.items() if n >= min_count]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return [word for word, _ in kept[:max_words]]


def encode(texts: Iterable[str], spec: TextSpec) -> np.ndarray:
    """`[rows, max_tokens]` int64 ids."""
    ids = spec.index
    rows = [[ids.get(t, UNKNOWN) for t in tokenize(text)[: spec.max_tokens]] for text in texts]
    out = np.full((len(rows), spec.max_tokens), PAD, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def clean(values: pd.Series) -> list[str]:
    """A frame column as strings: a missing text is the empty string, which encodes to all padding."""
    return values.fillna("").astype(str).tolist()


@dataclass
class Split:
    texts: list[str]
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def prepare(
    df: pd.DataFrame, *, text_column: str, label_column: str, class_names: list[str], split_column: str
) -> dict[str, Split]:
    """The usable rows of each snapshot split ("train", "validation", "test"): a row needs a known class and a text with
    something in it. Labels are the class's index in `class_names`."""
    index = {name: i for i, name in enumerate(class_names)}
    df = df.reset_index(drop=True)
    labels = df[label_column].map(lambda v: index.get(v) if isinstance(v, str) else None)
    texts = df[text_column]
    usable = labels.notna() & texts.notna() & texts.astype(str).str.strip().ne("")
    out: dict[str, Split] = {}
    for name in ("train", "validation", "test"):
        rows = df[usable & (df[split_column] == name)]
        out[name] = Split(texts=clean(rows[text_column]), y=labels[rows.index].to_numpy(dtype=np.int64))
    return out
