"""The headline score of a run is read off its evaluation report, per task."""

import uuid

import pytest

from omnitrain.db.models import RunEvaluation
from omnitrain.schemas.training import EvaluationScore
from omnitrain.services.evaluation_score import best_score, evaluation_brief, primary_score


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


def _score(metric="accuracy", value=0.5, higher=True) -> EvaluationScore:
    return EvaluationScore(metric=metric, label=metric, value=value, higher_is_better=higher)


class TestBestScore:
    def test_the_best_of_a_projects_runs_wins(self):
        best = best_score([_score(value=0.7), _score(value=0.9), _score(value=0.8)])
        assert best is not None and best.value == 0.9

    def test_for_an_error_metric_lower_is_better(self):
        best = best_score([_score("rmse", 3.0, False), _score("rmse", 1.5, False), _score("rmse", 2.0, False)])
        assert best is not None and best.value == 1.5

    def test_runs_judged_by_another_metric_than_the_newest_are_not_compared_with_it(self):
        # Newest first: a detector's mAP50, then two older classifier runs whose accuracy is higher.
        best = best_score([_score("map50", 0.4), _score("accuracy", 0.99), _score("accuracy", 0.95)])
        assert best is not None and (best.metric, best.value) == ("map50", 0.4)

    def test_no_scored_run_means_no_best(self):
        assert best_score([]) is None


class TestScorePlugins:
    @pytest.fixture
    def scores(self):
        from omnitrain.scoring.base import _registry

        _registry.ensure_loaded()
        before = set(_registry._items)
        yield _registry
        for plugin_id in set(_registry._items) - before:
            _registry.unregister(plugin_id)

    def test_the_builtin_scores_are_tried_most_telling_first(self):
        from omnitrain.scoring import list_scores

        assert [s.id for s in list_scores()] == ["entityF1", "accuracy", "map50", "r2", "rmse", "mae", "rougeL"]

    def test_a_new_class_becomes_a_headline_with_no_other_edit(self, scores):
        from omnitrain.scoring import HeadlineScore

        class Ndcg(HeadlineScore):
            id = "ndcg"
            label = "NDCG@10"
            order = 15

        score = primary_score(_evaluation(report={"overall": {"ndcg": 0.62, "r2": 0.4}}))
        assert score is not None and (score.metric, score.label, score.value) == ("ndcg", "NDCG@10", 0.62)
        # A built-in with a lower `order` still wins when the report carries it.
        score = primary_score(_evaluation(accuracy=0.9, report={"overall": {"ndcg": 0.62}}))
        assert score is not None and score.metric == "accuracy"

    def test_a_score_can_read_its_value_from_somewhere_else(self, scores):
        from omnitrain.scoring import HeadlineScore

        class Nested(HeadlineScore):
            id = "bleu"
            label = "BLEU"

            @classmethod
            def value(cls, overall):
                raw = overall.get("text", {}).get("bleu")
                return float(raw) if raw is not None else None

        score = primary_score(_evaluation(report={"overall": {"text": {"bleu": 0.3}}}))
        assert score is not None and (score.metric, score.value) == ("bleu", 0.3)

    def test_a_score_without_a_label_is_refused_and_so_is_a_duplicate_id(self, scores):
        from omnitrain.scoring import HeadlineScore

        with pytest.raises(TypeError, match="needs a label"):

            class Unlabelled(HeadlineScore):
                id = "unlabelled"

        with pytest.raises(ValueError, match="Duplicate headline score id"):

            class Again(HeadlineScore):
                id = "accuracy"
                label = "Accuracy again"
