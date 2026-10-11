"""The headline score of a run is read off its evaluation report, per task."""

import uuid

from omnitrain.db.models import RunEvaluation
from omnitrain.services.evaluation_score import evaluation_brief, primary_score


def _evaluation(status="success", accuracy=None, report=None) -> RunEvaluation:
    return RunEvaluation(run_id=uuid.uuid4(), status=status, accuracy=accuracy, macro_f1=None, report=report)


def test_a_classifier_is_scored_by_accuracy():
    score = primary_score(_evaluation(accuracy=0.9, report={"overall": {"accuracy": 0.9, "macroF1": 0.8}}))
    assert score is not None
    assert (score.metric, score.label, score.higher_is_better) == ("accuracy", "Accuracy", True)
    assert score.value == 0.9


def test_a_detector_is_scored_by_map50_not_a_missing_accuracy():
    report = {"overall": {"precision": 0.7, "recall": 0.6, "map50": 0.55, "map5095": 0.31}}
    score = primary_score(_evaluation(report=report))
    assert score is not None and (score.metric, score.label, score.value) == ("map50", "mAP50", 0.55)


def test_a_regressor_prefers_r2_and_falls_back_to_an_error_where_lower_is_better():
    r2 = primary_score(_evaluation(report={"overall": {"mae": 2.0, "rmse": 3.0, "r2": 0.4}}))
    assert r2 is not None and r2.metric == "r2" and r2.higher_is_better
    rmse = primary_score(_evaluation(report={"overall": {"mae": 2.0, "rmse": 3.0, "r2": None}}))
    assert rmse is not None and rmse.metric == "rmse" and not rmse.higher_is_better


def test_no_recognised_metric_means_no_score():
    assert primary_score(_evaluation(report={"overall": {"bleu": 0.2}})) is None
    assert primary_score(_evaluation(report=None)) is None


def test_a_failed_evaluation_carries_no_score():
    brief = evaluation_brief(_evaluation(status="failed", accuracy=0.9, report={"overall": {"accuracy": 0.9}}))
    assert brief is not None and brief.score is None
    assert evaluation_brief(None) is None
