"""The Ludwig half of training: build a `LudwigModel`, run `.train()`, stream progress.

Moved out of jobs/train.py when trainer backends became a plugin system. What changed relative to
the legacy NATS worker still applies here: progress goes straight to the event writer via
`TrainContext.report` (a thread-safe queue put), and abort is a check at epoch boundaries (Ludwig
exposes no finer hook), not a JetStream KV read.
"""

import logging
from typing import Any

from ludwig.api import LudwigModel
from ludwig.callbacks import Callback
from opentelemetry import trace

from theseus.backends.base import TrainContext
from theseus.backends.ludwig.model import LudwigLoadedModel

tracer = trace.get_tracer("theseus")


def _extract_metrics(feature_metrics: dict[str, dict[str, list[Any]]]) -> dict[str, float]:
    """Flatten every metric Ludwig tracks for a split into {metricName: latest value}.

    Per-output-feature metrics are namespaced `{featureName}.{metric}` so a multi-output config
    cannot collide; the `combined` feature (Ludwig's aggregate across outputs) stays unprefixed.
    """
    import ludwig.constants as ludwig_consts

    out: dict[str, float] = {}
    for feature_name, per_metric in feature_metrics.items():
        prefix = "" if feature_name == ludwig_consts.COMBINED else f"{feature_name}."
        for metric_name, history in per_metric.items():
            if history:
                out[f"{prefix}{metric_name}"] = float(history[-1].value)
    return out


class TrainingProgressCallback(Callback):
    """Ludwig callback (runs in the training thread): heartbeat, abort check, per-epoch metrics."""

    def __init__(self, run: TrainContext) -> None:
        self.run = run

    def on_epoch_start(self, trainer, progress_tracker, save_path, **kwargs):
        # Epoch-end events can be minutes apart on large datasets; this keeps the run looking alive.
        self.run.check_abort()
        self.run.heartbeat()

    def on_epoch_end(self, trainer, progress_tracker, save_path, **kwargs):
        self.run.check_abort()
        epoch = progress_tracker.epoch
        trace.get_current_span().add_event(f"epoch {epoch} end", {"epoch": epoch})
        splits = {
            "train": progress_tracker.train_metrics,
            "validation": progress_tracker.validation_metrics,
            "test": progress_tracker.test_metrics,
        }
        for split_name, split_metrics in splits.items():
            metrics = _extract_metrics(split_metrics)
            if metrics:
                self.run.report(epoch, split_name, metrics)


def train(run: TrainContext) -> LudwigLoadedModel:
    model = LudwigModel(config=run.config, logging_level=logging.INFO, callbacks=[TrainingProgressCallback(run)])
    with tracer.start_as_current_span("ludwig.train"):
        model.train(dataset=run.dataset_uri, output_directory=run.output_uri, experiment_name="results")
    return LudwigLoadedModel(model)
