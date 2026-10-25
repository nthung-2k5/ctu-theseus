"""Build the scikit-learn estimator a compiled config describes, and save and load a trained one.

A model is stored with skops, not pickle: loading a pickle runs whatever code is inside it, and a model trained on a
remote worker is loaded on the API host. skops refuses any type it has not been told to trust, so only scikit-learn's
own classes can come out of a file. `TRUSTED` is the one exception, reviewed below.
"""

from pathlib import Path
from typing import Any

from omnitrain.plugins.backends.scikit import tasks

MODEL_FILENAME = "model.skops"
METADATA_FILENAME = "names.json"

# Types skops reports as untrusted in the models this backend produces, trusted here after review:
#  - the node tables of a gradient-boosted tree and of a decision tree (a forest): a forged file could hold out-of-range
#    node indices, which can crash the process when it predicts, but cannot run code;
#  - `numpy.dtype`, which the imputer and scaler keep as an attribute; arrays themselves are stored without pickle.
# Anything else in a file, a class that is not scikit-learn's own above all, is refused. The files are only ever written
# by this backend's own training.
TRUSTED = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def build_estimator(config: dict[str, Any]) -> Any:
    """An unfitted pipeline: the preprocessing the model needs, then the model."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model_id: str = config["model"]
    classify = config["task"] == "classify"
    params: dict[str, Any] = config["params"]
    seed = config.get("seed")
    class_weight = params.get("classWeight")

    if model_id == tasks.HIST_GRADIENT_BOOSTING:
        options: dict[str, Any] = {
            "max_iter": params.get("nEstimators", 100),
            "max_depth": params.get("maxDepth"),
            "learning_rate": params.get("learningRate", 0.1),
            # The snapshot already has a validation split; a hidden holdout would shrink the training data.
            "early_stopping": False,
            "random_state": seed,
        }
        if classify:
            options["class_weight"] = class_weight
            return HistGradientBoostingClassifier(**options)
        return HistGradientBoostingRegressor(**options)

    if model_id == tasks.RANDOM_FOREST:
        options = {
            "n_estimators": params.get("nEstimators", 200),
            "max_depth": params.get("maxDepth"),
            "n_jobs": -1,
            "random_state": seed,
        }
        if classify:
            # Imputed so that a missing value at serving time is scored, not an error.
            forest = RandomForestClassifier(class_weight=class_weight, **options)
        else:
            forest = RandomForestRegressor(**options)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", forest)])

    if model_id == tasks.LOGISTIC_REGRESSION and "text" not in config:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params.get("c", 1.0), class_weight=class_weight, max_iter=1000, random_state=seed
                    ),
                ),
            ]
        )

    if model_id == tasks.RIDGE:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=params.get("alpha", 1.0), random_state=seed)),
            ]
        )

    text = config["text"]
    vectorizer = TfidfVectorizer(
        analyzer=text["analyzer"],
        ngram_range=(1, text["ngramMax"]),
        max_features=text["maxFeatures"],
        min_df=text["minDf"],
        sublinear_tf=text["sublinearTf"],
    )
    if model_id == tasks.NAIVE_BAYES:
        return Pipeline([("tfidf", vectorizer), ("model", MultinomialNB(alpha=params.get("alpha", 1.0)))])
    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "model",
                LogisticRegression(C=params.get("c", 1.0), class_weight=class_weight, max_iter=1000, random_state=seed),
            ),
        ]
    )


def save(estimator: Any, directory: str) -> str:
    import skops.io

    path = str(Path(directory) / MODEL_FILENAME)
    skops.io.dump(estimator, path)
    return path


def load(directory: str) -> Any:
    import skops.io

    return skops.io.load(str(Path(directory) / MODEL_FILENAME), trusted=TRUSTED)


def final_estimator(estimator: Any) -> Any:
    """The model itself, whether or not it sits at the end of a pipeline."""
    steps = getattr(estimator, "steps", None)
    return steps[-1][1] if steps else estimator
