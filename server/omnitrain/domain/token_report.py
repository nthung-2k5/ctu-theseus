"""The evaluation report for a model that gives one tag per word, shared by every backend that does.

It has the classification report's shape (the web panel and `run_evaluations` read it without knowing the task), counted
over WORDS: `confusionMatrix` and `perClass` say, for each tag, how many words were tagged right. A model that says `O`
everywhere scores well on words, so when the tags are in the BIO convention (`B-PER`, `I-PER`, `O`) `overall` also holds
entity-level precision, recall and F1: an entity counts only when its type and both ends are exactly right (CoNLL's
`conlleval`, which `seqeval` reproduces). That is the headline (`scoring/metrics.py`).

`model` is any `LoadedModel` whose predictions follow `{feature}_predictions` (the tags joined by single spaces, as the
snapshot's tags cell is) and `{feature}_probability`.
"""

from typing import Any

import numpy as np
import pandas as pd

from omnitrain.domain.evaluate import pick_eval_split, top_errors
from omnitrain.domain.token_tags import cell_tags, words
from omnitrain.plugins.backends.base import EvalResult, LoadedModel

Entity = tuple[str, int, int]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def entities(tags: list[str]) -> set[Entity]:
    """The entities in a BIO tag sequence as `(type, first word, last word)`. `B-X` starts one, `I-X` continues the one
    that is open when it has type X and otherwise starts a new one (a stray `I-` is not dropped); `O` and anything
    that is not `B-`/`I-` closes it."""
    found: set[Entity] = set()
    start, kind = -1, ""
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and label == kind and start >= 0
        if start >= 0 and not continues:
            found.add((kind, start, i - 1))
            start, kind = -1, ""
        if prefix in ("B", "I") and label and start < 0:
            start, kind = i, label
    return found


def is_bio(labels: list[str]) -> bool:
    """True when the tag set has entity tags: some `B-x` or `I-x`."""
    return any(label[:2] in ("B-", "I-") and len(label) > 2 for label in labels)


def evaluate_tokens(
    model: LoadedModel,
    df: pd.DataFrame,
    split_column: str,
    item_id_column: str,
    *,
    label_column: str,
    batch_size: int,
    max_classes_for_matrix: int,
) -> EvalResult | None:
    labels = list(model.output.labels or [])
    inputs = list(model.input_columns)
    name = model.output.name
    index = {label: i for i, label in enumerate(labels)}

    eval_rows, split_used = pick_eval_split(df, split_column)
    usable = eval_rows[inputs].notna().all(axis=1) & eval_rows[label_column].notna()
    eval_rows = eval_rows[usable].reset_index(drop=True)
    # A row is scored when it has one tag per word and every tag is one this model has heard of.
    texts = eval_rows[inputs[0]].astype(str).tolist()
    actual_tags = [cell_tags(cell) for cell in eval_rows[label_column]]
    keep = [
        len(tags) == len(words(text)) > 0 and all(tag in index for tag in tags)
        for text, tags in zip(texts, actual_tags, strict=True)
    ]
    eval_rows = eval_rows[keep].reset_index(drop=True)
    actual_tags = [tags for tags, ok in zip(actual_tags, keep, strict=True) if ok]
    if len(eval_rows) == 0:
        return None

    batches = [
        model.predict(eval_rows.iloc[start : start + batch_size][inputs])
        for start in range(0, len(eval_rows), batch_size)
    ]
    predictions = pd.concat(batches, ignore_index=True)
    predicted_tags = [cell_tags(cell) for cell in predictions[f"{name}_predictions"]]

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    rows = [(index[a], index[p]) for actual, predicted in zip(actual_tags, predicted_tags, strict=True)
            for a, p in zip(actual, predicted, strict=True)]  # fmt: skip
    np.add.at(matrix, ([a for a, _ in rows], [p for _, p in rows]), 1)

    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    per_class: dict[str, dict[str, Any]] = {}
    f1_scores: list[float] = []
    for i, label in enumerate(labels):
        precision = _ratio(true_positive[i], predicted_count[i])
        recall = _ratio(true_positive[i], support[i])
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support[i]),
        }
        if support[i] or predicted_count[i]:
            f1_scores.append(f1)

    overall: dict[str, Any] = {
        "accuracy": round(_ratio(true_positive.sum(), matrix.sum()), 4),
        "macroF1": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
    }
    if is_bio(labels):
        correct = wanted = found = 0
        for actual, predicted in zip(actual_tags, predicted_tags, strict=True):
            real, guessed = entities(actual), entities(predicted)
            correct += len(real & guessed)
            wanted += len(real)
            found += len(guessed)
        precision, recall = _ratio(correct, found), _ratio(correct, wanted)
        overall["entityPrecision"] = round(precision, 4)
        overall["entityRecall"] = round(recall, 4)
        overall["entityF1"] = round(_ratio(2 * precision * recall, precision + recall), 4)

    truncated = len(labels) > max_classes_for_matrix
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "split": split_used,
        "outputFeature": name,
        "outputType": "tags",
        "rowCount": len(eval_rows),
        # What the matrix and the per-tag numbers count: words, not rows.
        "unit": "word",
        "wordCount": int(matrix.sum()),
        "idx2str": labels,
        "truncated": truncated,
        "overall": overall,
        # A row is an error when any of its tags differs; the most confident wrong sentences come first.
        "topErrors": top_errors(eval_rows, predictions, label_column, name, item_id_column),
    }
    if not truncated:
        report["confusionMatrix"] = matrix.tolist()
        report["perClass"] = per_class

    download = predictions.drop(columns=[f"{name}_words", f"{name}_word_probabilities"], errors="ignore")
    if item_id_column in eval_rows.columns:
        download[item_id_column] = eval_rows[item_id_column].values
    download[f"{name}_actual"] = eval_rows[label_column].values
    return EvalResult(report=report, predictions=download)
