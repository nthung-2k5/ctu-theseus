"""How a generated text is scored against the written one (pure Python, no torch, no extra package).

Every score is a number in 0..1 where higher is better. Texts are compared as lower-cased words with punctuation
dropped, split on whitespace. That is a deliberately small definition: it works for any language that separates words
with spaces (Vietnamese included), and takes no dictionary of English articles or stemmers, so a score here is not
comparable with a published ROUGE or SQuAD figure, only with another run of this backend.
"""

import unicodedata
from collections import Counter
from collections.abc import Sequence


def words(text: str) -> list[str]:
    """Lower-cased words; punctuation and symbols are blanks."""
    folded = unicodedata.normalize("NFC", text).lower()
    return "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in folded).split()


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 and reference == 0:
        return 1.0  # both are empty, so they agree
    if overlap == 0:
        return 0.0
    precision, recall = overlap / predicted, overlap / reference
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(words(prediction) == words(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Overlap of the two texts' words, counting a word as often as both have it."""
    p, r = words(prediction), words(reference)
    return _f1(sum((Counter(p) & Counter(r)).values()), len(p), len(r))


def rouge_n(prediction: str, reference: str, n: int) -> float:
    """F1 of the n-word sequences the two texts share (ROUGE-1 with n=1, ROUGE-2 with n=2)."""
    p, r = words(prediction), words(reference)
    p_grams = Counter(tuple(p[i : i + n]) for i in range(len(p) - n + 1))
    r_grams = Counter(tuple(r[i : i + n]) for i in range(len(r) - n + 1))
    if not p_grams and not r_grams:
        # Both texts are shorter than n words (a one-word answer has no bigram): "no n-grams on either side" does not
        # mean they agree, so the texts themselves decide.
        return float(p == r)
    return _f1(sum((p_grams & r_grams).values()), sum(p_grams.values()), sum(r_grams.values()))


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """F1 of the longest run of words the two texts share in the same order (not necessarily next to each other)."""
    p, r = words(prediction), words(reference)
    return _f1(_lcs(p, r), len(p), len(r))


def score(prediction: str, reference: str) -> dict[str, float]:
    return {
        "exactMatch": exact_match(prediction, reference),
        "tokenF1": token_f1(prediction, reference),
        "rouge1": rouge_n(prediction, reference, 1),
        "rouge2": rouge_n(prediction, reference, 2),
        "rougeL": rouge_l(prediction, reference),
    }
