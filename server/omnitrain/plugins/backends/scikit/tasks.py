"""The tasks the scikit-learn backend trains and the model families it offers for each. Pure data: nothing here
imports scikit-learn, so listing the backend never loads it."""

from dataclasses import dataclass
from typing import Literal

from omnitrain.plugins.backends.base import ModelChoice


@dataclass(frozen=True)
class SklearnTaskSpec:
    # "classify": one class per row. "regress": one number per row.
    kind: Literal["classify", "regress"]
    # "record": the scalar columns of the snapshot. "text": the one inline text column.
    input: Literal["record", "text"]
    models: list[ModelChoice]


# Model ids, shared by tasks that use the same family.
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
RANDOM_FOREST = "random_forest"
LOGISTIC_REGRESSION = "logistic_regression"
RIDGE = "ridge"
NAIVE_BAYES = "naive_bayes"

_BOOSTING = ModelChoice(
    id=HIST_GRADIENT_BOOSTING,
    label="Gradient-boosted trees",
    description="Many small decision trees, each correcting the last. Usually the most accurate choice for a table, "
    "and it copes with missing values.",
)
_FOREST = ModelChoice(
    id=RANDOM_FOREST,
    label="Random forest",
    description="Many decision trees voting. Hard to overfit and needs little tuning; a little less accurate than "
    "boosting.",
)

SKLEARN_TASKS: dict[str, SklearnTaskSpec] = {
    "tabular_classification": SklearnTaskSpec(
        kind="classify",
        input="record",
        models=[
            _BOOSTING,
            _FOREST,
            ModelChoice(
                id=LOGISTIC_REGRESSION,
                label="Logistic regression",
                description="A straight-line boundary between classes. Fast and easy to interpret, but it cannot "
                "learn interactions between columns.",
            ),
        ],
    ),
    "tabular_regression": SklearnTaskSpec(
        kind="regress",
        input="record",
        models=[
            _BOOSTING,
            _FOREST,
            ModelChoice(
                id=RIDGE,
                label="Ridge regression",
                description="A straight-line fit with a penalty on large weights. Fast and easy to interpret, but it "
                "cannot learn interactions between columns.",
            ),
        ],
    ),
    "text_classification": SklearnTaskSpec(
        kind="classify",
        input="text",
        models=[
            ModelChoice(
                id=LOGISTIC_REGRESSION,
                label="TF-IDF + logistic regression",
                description="Counts the words and short phrases in each text, weighting rare ones higher, then fits "
                "a linear classifier. A fast baseline that trains in seconds on a CPU.",
            ),
            ModelChoice(
                id=NAIVE_BAYES,
                label="TF-IDF + naive Bayes",
                description="The same word counts with a probabilistic classifier. Even faster, and often good on "
                "small datasets, but usually a little less accurate.",
            ),
        ],
    ),
}

ANALYZERS = ("word", "char_wb")
MAX_FEATURES_CHOICES = (5_000, 20_000, 50_000, 200_000)
DEFAULT_MAX_FEATURES = 50_000
DEFAULT_NGRAM_MAX = 2
