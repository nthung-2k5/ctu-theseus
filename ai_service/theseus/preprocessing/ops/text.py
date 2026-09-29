"""Text preprocessing. Samples are plain strings."""

import re
from typing import Any

from theseus.preprocessing.base import NoParams, Preprocessing

_WHITESPACE = re.compile(r"\s+")


class Lowercase(Preprocessing):
    id = "text_lowercase"
    label = "Lowercase"
    description = "Lowercase all text."
    modality = "text"
    order = 10

    @classmethod
    def apply(cls, sample: str, params: NoParams, state: Any = None) -> str:
        return sample.lower()


class NormalizeWhitespace(Preprocessing):
    id = "text_normalize_whitespace"
    label = "Normalize whitespace"
    description = "Collapse runs of whitespace to a single space and trim the ends."
    modality = "text"
    order = 20

    @classmethod
    def apply(cls, sample: str, params: NoParams, state: Any = None) -> str:
        return _WHITESPACE.sub(" ", sample).strip()
