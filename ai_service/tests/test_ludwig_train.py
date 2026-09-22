"""Ludwig-specific training glue: the progress/abort callback and _extract_metrics.

The generic engine around this (run lifecycle, DB status, log upload) is tested against a fake
backend in test_jobs_train.py; these tests are about the part that's actually Ludwig-shaped: a
`ludwig.callbacks.Callback` translating epoch/metric events into `TrainContext` calls.
"""

from types import SimpleNamespace

import pytest

from theseus.backends.base import TrainContext
from theseus.backends.ludwig import train as train_mod
from theseus.backends.ludwig.model import LudwigLoadedModel
from theseus.jobs import abort


class Metric:
    def __init__(self, value):
        self.value = value


def tracker(epoch, val_loss):
    return SimpleNamespace(
        epoch=epoch,
        train_metrics={"combined": {"loss": [Metric(val_loss + 0.1)]}, "class": {"accuracy": [Metric(0.5)]}},
        validation_metrics={"combined": {"loss": [Metric(val_loss)]}},
        test_metrics={},
    )


class FakeModel:
    """Stands in for LudwigModel: runs epochs through the REAL callback."""

    instances: list["FakeModel"] = []
    epochs = 3
    losses = [0.9, 0.4, 0.6]
    fail_with: Exception | None = None

    def __init__(self, config, logging_level=None, callbacks=None):
        self.config, self.callbacks = config, callbacks or []
        self.config_obj = SimpleNamespace(
            input_features=[SimpleNamespace(column="x")],
            output_features=[SimpleNamespace(name="class", type="category", column="class")],
        )
        self.training_set_metadata = {"class": {"idx2str": ["a", "b"]}}
        self.trained_epochs = 0
        FakeModel.instances.append(self)

    def train(self, dataset, output_directory, experiment_name):
        for epoch in range(1, self.epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(None, tracker(epoch, self.losses[epoch - 1]), None)
            if self.fail_with:
                raise self.fail_with
            for cb in self.callbacks:
                cb.on_epoch_end(None, tracker(epoch, self.losses[epoch - 1]), None)
            self.trained_epochs = epoch
        return "trained"


def _ctx(run_id="r1", **overrides):
    calls = SimpleNamespace(heartbeats=0, aborts=0, reports=[])

    def heartbeat():
        calls.heartbeats += 1

    def check_abort():
        calls.aborts += 1
        overrides.get("check_abort_hook", lambda: None)()

    def report(epoch, split, metrics):
        calls.reports.append((epoch, split, metrics))

    ctx = TrainContext(
        run_id=run_id, config={}, dataset_uri="d", output_uri="o", workdir="w",
        _heartbeat=heartbeat, _check_abort=check_abort, _report=report,
    )  # fmt: skip
    return ctx, calls


@pytest.fixture(autouse=True)
def fake_ludwig_model(monkeypatch):
    FakeModel.instances = []
    FakeModel.epochs, FakeModel.losses, FakeModel.fail_with = 3, [0.9, 0.4, 0.6], None
    monkeypatch.setattr(train_mod, "LudwigModel", FakeModel)


def test_train_runs_every_epoch_reports_metrics_and_returns_a_loaded_model():
    ctx, calls = _ctx()

    result = train_mod.train(ctx)

    assert isinstance(result, LudwigLoadedModel)
    assert calls.heartbeats == 3  # once per epoch, from on_epoch_start
    assert calls.aborts == 6  # twice per epoch: on_epoch_start and on_epoch_end both check
    # namespaced per feature except combined, one train + one validation report per epoch
    assert calls.reports[0] == (1, "train", {"loss": pytest.approx(1.0), "class.accuracy": 0.5})
    assert calls.reports[1] == (1, "validation", {"loss": pytest.approx(0.9)})
    assert len(calls.reports) == 6


def test_check_abort_raising_stops_training_immediately():
    # The first check_abort call (epoch 1's on_epoch_start) is free; make the second
    # (on_epoch_end, same epoch) raise, mimicking a cancel landing mid-epoch.
    seen = {"n": 0}

    def check_abort_hook():
        seen["n"] += 1
        if seen["n"] >= 2:
            raise abort.TrainingAborted("stop")

    ctx, calls = _ctx(check_abort_hook=check_abort_hook)

    with pytest.raises(abort.TrainingAborted):
        train_mod.train(ctx)

    assert FakeModel.instances[0].trained_epochs == 0  # aborted before epoch 1 finished


def test_metric_extraction_prefixes_output_features_but_not_the_combined_aggregate():
    out = train_mod._extract_metrics(
        {"combined": {"loss": [Metric(0.1), Metric(0.2)]}, "label": {"accuracy": [Metric(0.9)], "empty": []}}
    )
    assert out == {"loss": 0.2, "label.accuracy": 0.9}  # latest value; empty history skipped
