"""The scikit-learn half of training: read the snapshot, fit once, report, upload.

A fit has no epochs, so it is reported as epoch 1: the train and validation loss (log loss for a classifier, mean
squared error for a regressor) let `events/writer.py` mark that epoch as the run's best. There is no abort check inside
`fit()`, which scikit-learn does not allow; the run is checked before and after it, and a fit takes seconds to minutes.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from opentelemetry import trace

from omnitrain.core import constants as C
from omnitrain.plugins.backends.base import ConfigError, TrainContext
from omnitrain.plugins.backends.scikit import data, estimators
from omnitrain.plugins.backends.scikit.model import SklearnLoadedModel
from omnitrain.plugins.backends.scikit.model import load as load_model
from omnitrain.services import storage

tracer = trace.get_tracer("omnitrain")


def _scores(estimator: Any, split: data.Split, kind: str) -> dict[str, float]:
    """Loss and the headline metric of a fitted model on one split. A class the model never saw in training cannot be
    scored, so those rows are left out of a classifier's numbers."""
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score

    if kind == "regress":
        predicted = estimator.predict(split.x)
        scores = {
            "loss": float(mean_squared_error(split.y, predicted)),
            "mean_absolute_error": float(mean_absolute_error(split.y, predicted)),
        }
        if len(split) > 1:
            scores["r2"] = float(r2_score(split.y, predicted))
        return scores

    classes = list(estimators.final_estimator(estimator).classes_)
    known = split.y.isin(classes).to_numpy()
    if not known.any():
        return {}
    x = split.x[known]
    y = split.y[known]
    probabilities = estimator.predict_proba(x)
    return {
        "loss": float(log_loss(y, probabilities, labels=classes)),
        "accuracy": float(accuracy_score(y, estimator.predict(x))),
    }


def train(run: TrainContext) -> SklearnLoadedModel:
    cfg = run.config
    workdir = Path(run.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        df = pd.read_parquet(run.dataset_uri)
        run.heartbeat()
        splits = data.prepare(df, cfg, C.SPLIT_COLUMN_NAME)
        train_split = splits["train"]
        if len(train_split) == 0:
            raise ConfigError("The train split has no labelled rows, which scikit-learn needs to fit a model")
        kind: str = cfg["task"]
        if kind == "classify" and train_split.y.nunique() < 2:
            raise ConfigError("The train split has only one class, so there is nothing to tell apart")
        run.check_abort()

        estimator = estimators.build_estimator(cfg)
        with tracer.start_as_current_span("sklearn.fit"):
            estimator.fit(train_split.x, train_split.y.to_numpy() if kind == "regress" else train_split.y)
        run.check_abort()

        for name in ("train", "validation"):
            if len(splits[name]):
                scores = _scores(estimator, splits[name], kind)
                if scores:
                    run.report(1, name, scores)

        meta: dict[str, Any] = {
            "schemaVersion": 1,
            "task": kind,
            "model": cfg["model"],
            "inputKind": "text" if "text" in cfg else "record",
            "inputColumns": cfg["inputColumns"],
            # Also the prediction frame's column prefix (see model.py).
            "outputName": cfg["labelColumn"],
            # The model's own class order, from the fitted `classes_`. Index i of every probability list is names[i].
            "names": [str(c) for c in estimators.final_estimator(estimator).classes_] if kind == "classify" else [],
        }
        stage = workdir / "upload"
        stage.mkdir()
        estimators.save(estimator, str(stage))
        (stage / estimators.METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        storage.upload_dir(C.BUCKET_TRAINING, storage.training_results_prefix(run.run_id), str(stage))

        # Loaded from what was just uploaded, so serving and evaluation see exactly the same files.
        return load_model(str(stage))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
