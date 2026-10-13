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


def primary_score(evaluation: RunEvaluation) -> EvaluationScore | None:
    overall: dict[str, Any] = {}
    if isinstance(evaluation.report, dict) and isinstance(evaluation.report.get("overall"), dict):
        overall = evaluation.report["overall"]
    if evaluation.accuracy is not None:
        overall = {**overall, "accuracy": evaluation.accuracy}
    for score in list_scores():
        value = score.value(overall)
        if value is not None:
            return EvaluationScore(
                metric=score.id, label=score.label, value=value, higher_is_better=score.higher_is_better
            )
    return None


def evaluation_brief(evaluation: RunEvaluation | None) -> EvaluationBrief | None:
    if evaluation is None:
        return None
    return EvaluationBrief(
        status=evaluation.status,
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        score=primary_score(evaluation) if evaluation.status == "success" else None,
    )
