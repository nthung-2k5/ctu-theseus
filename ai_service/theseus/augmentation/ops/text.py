"""Text augmentations. Samples are plain strings; whitespace between words is preserved."""

import random
import re
from typing import Any

from pydantic import Field

from theseus.augmentation.base import Augmentation, ParamsModel

_SPLIT = re.compile(r"(\s+)")


def _words(text: str) -> tuple[list[str], list[int]]:
    """(all tokens incl. whitespace, indices of the word tokens)."""
    tokens = _SPLIT.split(text)
    return tokens, [i for i, t in enumerate(tokens) if t and not t.isspace()]


class RandomWordDeletion(Augmentation):
    class Params(ParamsModel):
        delete_rate: float = Field(0.1, ge=0.02, le=0.5, title="Delete rate", description="Share of words removed.")

    id = "text_word_deletion"
    label = "Random word deletion"
    description = "Drop a random share of the words. At least one word is always kept."
    modality = "text"
    order = 10

    @classmethod
    def apply(cls, sample: str, params: Params, rng: random.Random, state: Any = None) -> str:
        tokens, idx = _words(sample)
        if len(idx) < 2:
            return sample
        drop = {i for i in idx if rng.random() < params.delete_rate}
        if len(drop) == len(idx):
            drop.discard(rng.choice(idx))
        kept = [t for i, t in enumerate(tokens) if i not in drop]
        return re.sub(r"\s+", " ", "".join(kept)).strip()


class RandomWordSwap(Augmentation):
    class Params(ParamsModel):
        swaps: int = Field(2, ge=1, le=10, title="Swaps", description="Number of random word pairs exchanged.")

    id = "text_word_swap"
    label = "Random word swap"
    description = "Exchange the positions of random pairs of words."
    modality = "text"
    order = 20

    @classmethod
    def apply(cls, sample: str, params: Params, rng: random.Random, state: Any = None) -> str:
        tokens, idx = _words(sample)
        if len(idx) < 2:
            return sample
        # Distinct positions for every swap, so a later swap can never undo an earlier one.
        pairs = min(params.swaps, len(idx) // 2)
        chosen = rng.sample(idx, 2 * pairs)
        for a, b in zip(chosen[::2], chosen[1::2], strict=True):
            tokens[a], tokens[b] = tokens[b], tokens[a]
        return "".join(tokens)


class RandomWordDuplication(Augmentation):
    class Params(ParamsModel):
        duplicate_rate: float = Field(0.05, ge=0.02, le=0.3, title="Duplicate rate")

    id = "text_word_duplication"
    label = "Random word duplication"
    description = "Repeat a random share of the words in place."
    modality = "text"
    order = 30

    @classmethod
    def apply(cls, sample: str, params: Params, rng: random.Random, state: Any = None) -> str:
        tokens, idx = _words(sample)
        if not idx:
            return sample
        picked = {i for i in idx if rng.random() < params.duplicate_rate} or {rng.choice(idx)}
        return "".join(f"{t} {t}" if i in picked else t for i, t in enumerate(tokens))


class CharacterTypos(Augmentation):
    class Params(ParamsModel):
        typo_rate: float = Field(
            0.02, ge=0.005, le=0.1, title="Typo rate", description="Share of letters hit by a typo.",
            json_schema_extra={"step": 0.005},
        )  # fmt: skip

    id = "text_typos"
    label = "Character typos"
    description = "Introduce typing mistakes: a letter dropped, doubled or swapped with its neighbour."
    modality = "text"
    order = 40

    @classmethod
    def apply(cls, sample: str, params: Params, rng: random.Random, state: Any = None) -> str:
        chars = list(sample)
        out: list[str] = []
        i = 0
        while i < len(chars):
            c = chars[i]
            if c.isalpha() and rng.random() < params.typo_rate:
                kind = rng.choice(("drop", "double", "swap"))
                if kind == "drop":
                    i += 1
                    continue
                if kind == "double":
                    out.extend((c, c))
                    i += 1
                    continue
                if i + 1 < len(chars) and chars[i + 1].isalpha():
                    out.extend((chars[i + 1], c))
                    i += 2
                    continue
            out.append(c)
            i += 1
        return "".join(out)
