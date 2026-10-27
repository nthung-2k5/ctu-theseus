"""Words and their tags: the data side of token classification, shared by training, evaluation and serving.

A text is cut into words by `str.split()` (`domain/token_tags.py`), and every word has one tag. The built-in tagger
reads each word by its id in a vocabulary fitted on the TRAIN texts only: the word with its ASCII capitals lower-cased
(nothing else is folded) is looked up exactly, `UNKNOWN` when absent, the first `max_tokens` words are kept and the
rest of the row is padded with `PAD`. Unlike the text classifier's tokenizer this one never splits a word further, so
"Hue," and "Hue" are two words: a tagging dataset separates punctuation itself. The ids are `text.py`'s.

numpy and pandas only: no torch.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from omnitrain.domain.token_tags import cell_tags, tag_set, words
from omnitrain.plugins.backends.pytorch.spec import TextSpec
from omnitrain.plugins.backends.pytorch.text import _LOWER, PAD, UNKNOWN

# The label of a position that is not a word, or a word the model is not asked about: ignored by the loss.
IGNORE = -100


def word_forms(text: str) -> list[str]:
    return [w.translate(_LOWER) for w in words(text)]


def fit_vocabulary(texts: Iterable[str], *, max_words: int, min_count: int) -> list[str]:
    """The `max_words` most frequent words (counting every occurrence) seen at least `min_count` times, most frequent
    first and alphabetical among equals, so the same texts always give the same vocabulary."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(word_forms(text))
    kept = [(word, n) for word, n in counts.items() if n >= min_count]
    kept.sort(key=lambda item: (-item[1], item[0]))
    return [word for word, _ in kept[:max_words]]


def encode(texts: Iterable[str], spec: TextSpec) -> np.ndarray:
    """`[rows, max_tokens]` int64 ids, one per word."""
    ids = spec.index
    rows = [[ids.get(w, UNKNOWN) for w in word_forms(text)[: spec.max_tokens]] for text in texts]
    out = np.full((len(rows), spec.max_tokens), PAD, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def label_ids(tag_rows: Sequence[Sequence[str]], tag_index: dict[str, int], max_tokens: int) -> np.ndarray:
    """`[rows, max_tokens]` int64 tag ids for the words the model reads; everything else is `IGNORE`."""
    out = np.full((len(tag_rows), max_tokens), IGNORE, dtype=np.int64)
    for i, row in enumerate(tag_rows):
        for j, tag in enumerate(row[:max_tokens]):
            out[i, j] = tag_index[tag]
    return out


def token_labels(
    positions: Sequence[Sequence[int]], tag_rows: Sequence[Sequence[str]], tag_index: dict[str, int], length: int
) -> np.ndarray:
    """`[rows, length]` int64 tag ids for a tokenizer that cuts words into pieces: a word's tag is put at the position
    of the token that speaks for it (`positions[row][word]`, `hf.word_positions`); every other position, and a word
    no token covers, is `IGNORE`."""
    out = np.full((len(tag_rows), length), IGNORE, dtype=np.int64)
    for i, (spoken, row) in enumerate(zip(positions, tag_rows, strict=True)):
        for position, tag in zip(spoken, row, strict=True):
            if position >= 0:
                out[i, position] = tag_index[tag]
    return out


def clean(values: pd.Series) -> list[str]:
    """A frame column as strings: a missing text is the empty string, which has no words."""
    return values.fillna("").astype(str).tolist()


@dataclass
class TagSplit:
    texts: list[str]
    tags: list[list[str]]

    def __len__(self) -> int:
        return len(self.texts)


def prepare(
    df: pd.DataFrame, *, text_column: str, tags_column: str, split_column: str
) -> tuple[dict[str, TagSplit], list[str]]:
    """The usable rows of each snapshot split, and the tag set of all of them.

    A row is usable when its tag cell holds exactly one tag per word of its text (the snapshot writes no cell for tags
    that do not fit, and a text with no words has nothing to tag). The tag set is every tag in the usable rows of every
    split, `O` first (`token_tags.tag_set`): it is what the model's output is indexed by, and a tag that only the test
    split has costs nothing but an output the model never learned."""
    df = df.reset_index(drop=True)
    texts = clean(df[text_column])
    cells = df[tags_column].tolist()
    usable = [
        len(words(text)) > 0 and isinstance(cell, str) and len(cell_tags(cell)) == len(words(text))
        for text, cell in zip(texts, cells, strict=True)
    ]
    names = tag_set(cell for cell, ok in zip(cells, usable, strict=True) if ok)
    out: dict[str, TagSplit] = {}
    split_names = df[split_column].tolist()
    for name in ("train", "validation", "test"):
        rows = [i for i, ok in enumerate(usable) if ok and split_names[i] == name]
        out[name] = TagSplit(texts=[texts[i] for i in rows], tags=[cell_tags(cells[i]) for i in rows])
    return out, names
