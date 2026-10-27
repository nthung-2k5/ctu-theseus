"""The evaluation report for a language model: how close its written answers are to the ones in the data.

There is no confusion matrix or class list to show, so the report is the `overall` block alone (the web lists it as it
lists any other): the mean of each text score over the evaluated rows, the loss on the written answers and the
perplexity that loss gives. The rows' predictions go in the accompanying download.
"""

import math
from typing import Any

import pandas as pd

from omnitrain.domain.evaluate import pick_eval_split
from omnitrain.plugins.backends.base import EvalResult
from omnitrain.plugins.backends.llm import metrics, prompt, tasks
from omnitrain.plugins.backends.llm.model import LlmLoadedModel

# exp() of a loss this large is beyond a float; a model this bad is "very large" either way.
_MAX_LOSS_FOR_PERPLEXITY = 50.0


def evaluate(model: LlmLoadedModel, df: pd.DataFrame, split_column: str, item_id_column: str) -> EvalResult | None:
    rows, split_used = pick_eval_split(df, split_column)
    column = model.output.name
    # A row with no written answer has nothing to compare with.
    rows = rows[[bool(prompt.format_answer(a)) for a in rows[column]]]
    if len(rows) == 0:
        return None
    total_rows = len(rows)
    rows = rows.head(tasks.EVAL_MAX_ROWS).reset_index(drop=True)

    predictions = model.predict(rows)
    answers = [prompt.format_answer(a) for a in rows[column]]
    scores = [metrics.score(p, a) for p, a in zip(predictions[model.column], answers, strict=True)]
    overall: dict[str, float] = {key: round(sum(s[key] for s in scores) / len(scores), 4) for key in scores[0]}
    loss_sum, tokens = model.answer_loss(rows)
    if tokens:
        mean_loss = loss_sum / tokens
        overall["loss"] = round(mean_loss, 4)
        overall["perplexity"] = round(math.exp(min(mean_loss, _MAX_LOSS_FOR_PERPLEXITY)), 4)

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "split": split_used,
        "outputFeature": column,
        "outputType": "text",
        # Generating is slow, so only the first rows are scored; `totalRows` says how many there were to choose from.
        "rowCount": len(rows),
        "totalRows": total_rows,
        "overall": overall,
    }
    download = predictions.copy()
    if item_id_column in rows.columns:
        download[item_id_column] = rows[item_id_column].values
    download[f"{column}_actual"] = answers
    return EvalResult(report=report, predictions=download)
