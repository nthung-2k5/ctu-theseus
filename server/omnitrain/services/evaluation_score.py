"""The one number that says how good a run is, whatever its task.

A classifier is judged by accuracy, a detector by mAP50, a regressor by R-squared. Which one applies is read off
the evaluation report's `overall` block, which each backend fills with the metrics that fit its output, so the
runs list, the "best run" marker and the comparison never hard-code accuracy. The metrics that can be the
headline are plugins (`omnitrain/scoring/`); the first one, by priority, that the report carries wins.
"""

from typing import Any

from omnitrain.db.models import RunEvaluation
from omnitrain.schemas.training import EvaluationBrief, EvaluationScore
from omnitrain.scoring import list_scores


def score_from_overall(overall: Any, accuracy: float | None = None) -> EvaluationScore | None:
    """The headline score of an evaluation report's `overall` block (plus the denormalized accuracy column)."""
    overall = dict(overall) if isinstance(overall, dict) else {}
    if accuracy is not None:
        overall = {**overall, "accuracy": accuracy}
    for score in list_scores():
        value = score.value(overall)
        if value is not None:
            return EvaluationScore(
                metric=score.id, label=score.label, value=value, higher_is_better=score.higher_is_better
            )
    return None


def primary_score(evaluation: RunEvaluation) -> EvaluationScore | None:
    overall = evaluation.report.get("overall") if isinstance(evaluation.report, dict) else None
    return score_from_overall(overall, evaluation.accuracy)


def best_score(newest_first: list[EvaluationScore]) -> EvaluationScore | None:
    """The best of a project's run scores. Runs of one project can be judged by different metrics (a classifier by
    accuracy, a detector by mAP50), and those do not compare, so only the newest run's metric is considered."""
    if not newest_first:
        return None
    metric = newest_first[0].metric
    same = [s for s in newest_first if s.metric == metric]
    return max(same, key=lambda s: s.value) if same[0].higher_is_better else min(same, key=lambda s: s.value)


def evaluation_brief(evaluation: RunEvaluation | None) -> EvaluationBrief | None:
    if evaluation is None:
        return None
    return EvaluationBrief(
        status=evaluation.status,
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        score=primary_score(evaluation) if evaluation.status == "success" else None,
    )
