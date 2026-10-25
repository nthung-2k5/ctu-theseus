"""Compile a task plus snapshot context plus hyperparameters into a scikit-learn training config.

The config is a plain dict stored verbatim as `training_runs.config`; `train.py` builds the estimator from it
(`estimators.py`). Everything that can be wrong with a request surfaces here as a `ConfigError`. Nothing here imports
scikit-learn.

Unlike the neural backends, a scikit-learn model is fitted in one go: there are no epochs, batches, learning-rate
schedule or early stopping. Those shared fields are refused rather than ignored. The knobs that belong to one model
family (trees, regularisation) are ignored by the others, so a sweep over the model choice and a knob still runs.
"""

from typing import Any, Literal

from pydantic import Field

from omnitrain.domain.task_registry import SnapshotContext, TaskDescriptor
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, HyperparamsBase, refuse_unsupported, seed_spec
from omnitrain.plugins.backends.scikit import tasks
from omnitrain.plugins.backends.scikit.tasks import SKLEARN_TASKS, SklearnTaskSpec
from omnitrain.schemas.common import ParamSpec

# Section headings of the hyperparameter form, in the order it shows them.
_MODEL = "Model details"
_TRAINING = "Training"
_DATA = "Data"

# What a neural backend trains with and a one-shot fit cannot use.
_NOT_APPLICABLE = (
    "epochs", "batch_size", "early_stop_patience", "weight_decay", "lr_schedule", "warmup_epochs", "grad_clip_norm",
    "mixed_precision",
)  # fmt: skip


class SklearnHyperparameters(HyperparamsBase):
    """User-facing hyperparameter choices (camelCase on the wire, and in training_runs.hyperparameters).

    `learning_rate` (from HyperparamsBase) is used by gradient-boosted trees only. None leaves a knob at the model's
    own default.
    """

    # Trees: the number of boosting rounds (gradient boosting) or of trees (random forest), and how deep each may grow.
    n_estimators: int | None = Field(default=None, ge=1, le=2000)
    max_depth: int | None = Field(default=None, ge=1, le=64)
    # Logistic regression's inverse regularisation strength (smaller is stronger), ridge's penalty and naive Bayes'
    # smoothing (larger is stronger).
    regularization_c: float | None = Field(default=None, gt=0, le=10_000)
    alpha: float | None = Field(default=None, gt=0, le=1_000)
    # Weight each class inversely to its frequency. Classification only.
    use_class_weights: bool | None = None
    # Text only.
    analyzer: Literal["word", "char_wb"] | None = None
    ngram_max: int | None = Field(default=None, ge=1, le=3)
    max_features: int | None = Field(default=None, ge=100, le=1_000_000)
    min_df: int | None = Field(default=None, ge=1, le=1000)
    sublinear_tf: bool | None = None


def _spec_for(task: TaskDescriptor) -> SklearnTaskSpec:
    spec = SKLEARN_TASKS.get(task.id)
    if spec is None:
        raise ConfigError(f'The scikit-learn backend does not support the task "{task.id}"')
    return spec


def _column(ctx: SnapshotContext, kind: str) -> str:
    name = next((c.name for c in ctx.columns if c.kind == kind), None)
    if name is None:
        raise ConfigError(f'The snapshot has no "{kind}" column, which the scikit-learn backend needs')
    return name


def compile_sklearn_config(
    task: TaskDescriptor, ctx: SnapshotContext, hp: SklearnHyperparameters, custom: CustomModelRef | None = None
) -> dict[str, Any]:
    if custom is not None:
        raise ConfigError("The scikit-learn backend does not support bring-your-own models")
    spec = _spec_for(task)
    refuse_unsupported(hp, "scikit-learn", *_NOT_APPLICABLE)
    if spec.kind == "classify" and len(ctx.label_class_names) < 2:
        raise ConfigError("Classification needs a snapshot with at least two classes")

    model_id = hp.model_id or spec.models[0].id
    if model_id not in {m.id for m in spec.models}:
        raise ConfigError(f'"{model_id}" is not a scikit-learn model for the task "{task.id}"')

    if spec.input == "text":
        inputs = [_column(ctx, "inline_text")]
    else:
        # Tabular tasks do not know their column names statically: every scalar column of the snapshot is a feature.
        inputs = [c.name for c in ctx.columns if c.kind == "scalar"]
        if not inputs:
            raise ConfigError(f'Task "{task.id}": the snapshot has no numeric columns to learn from')

    params: dict[str, Any] = {}
    if model_id in (tasks.HIST_GRADIENT_BOOSTING, tasks.RANDOM_FOREST):
        for key, value in (("nEstimators", hp.n_estimators), ("maxDepth", hp.max_depth)):
            if value is not None:
                params[key] = value
    if model_id == tasks.HIST_GRADIENT_BOOSTING and hp.learning_rate is not None:
        params["learningRate"] = hp.learning_rate
    if model_id == tasks.LOGISTIC_REGRESSION and hp.regularization_c is not None:
        params["c"] = hp.regularization_c
    if model_id in (tasks.RIDGE, tasks.NAIVE_BAYES) and hp.alpha is not None:
        params["alpha"] = hp.alpha
    if spec.kind == "classify" and hp.use_class_weights and model_id != tasks.NAIVE_BAYES:
        # Naive Bayes has no class weights; it learns class priors from the data instead.
        params["classWeight"] = "balanced"

    config: dict[str, Any] = {
        "backend": "sklearn",
        "task": spec.kind,
        "model": model_id,
        "inputColumns": inputs,
        "labelColumn": _column(ctx, "label"),
        # Informational: the class order a trained model uses is its own (`classes_`), never this.
        "classNames": list(ctx.label_class_names),
        "params": params,
    }
    if hp.seed is not None:
        config["seed"] = hp.seed
    if spec.input == "text":
        config["text"] = {
            "analyzer": hp.analyzer or "word",
            "ngramMax": hp.ngram_max or tasks.DEFAULT_NGRAM_MAX,
            "maxFeatures": hp.max_features or tasks.DEFAULT_MAX_FEATURES,
            "minDf": hp.min_df or 1,
            "sublinearTf": True if hp.sublinear_tf is None else hp.sublinear_tf,
        }
    return config


def hyperparameter_specs(task: TaskDescriptor) -> list[ParamSpec]:
    """The create-run/create-sweep form's knobs. Hand-written: a knob belongs to one model family and the neural
    backends' shared fields (epochs, batch size, ...) do not apply."""
    spec = SKLEARN_TASKS.get(task.id)
    if spec is None:
        return []
    model_ids = {m.id for m in spec.models}

    training: list[ParamSpec] = []
    if model_ids & {tasks.HIST_GRADIENT_BOOSTING, tasks.RANDOM_FOREST}:
        training += [
            ParamSpec(
                name="nEstimators",
                label="Number of trees",
                type="int",
                group=_MODEL,
                default=None,
                min=1,
                max=2000,
                step=10,
                description="How many decision trees to build (boosting rounds, for gradient-boosted trees). More "
                "trees fit better and take longer. Leave it unset for 100 (boosting) or 200 (forest). Trees only.",
            ),
            ParamSpec(
                name="maxDepth",
                label="Maximum tree depth",
                type="int",
                group=_MODEL,
                default=None,
                min=1,
                max=64,
                step=1,
                description="How many questions deep each tree may go. Shallower trees are simpler and less likely to "
                "memorise the examples. Leave it unset for no limit. Trees only.",
            ),
            ParamSpec(
                name="learningRate",
                label="Learning Rate",
                type="float",
                group=_MODEL,
                default=None,
                min=0.001,
                max=1,
                step=0.01,
                description="How much each new tree is allowed to change the answer. Lower is steadier but needs more "
                "trees. Leave it unset for 0.1. Gradient-boosted trees only.",
            ),
        ]
    if tasks.LOGISTIC_REGRESSION in model_ids:
        training.append(
            ParamSpec(
                name="regularizationC",
                label="Regularisation C",
                type="float",
                group=_MODEL,
                default=None,
                min=0.0001,
                max=10_000,
                step=0.1,
                description="How freely logistic regression may fit the examples. Smaller values keep the model "
                "simpler and guard against memorising; larger values fit more closely. Leave it unset for 1. "
                "Logistic regression only.",
            )
        )
    if model_ids & {tasks.RIDGE, tasks.NAIVE_BAYES}:
        training.append(
            ParamSpec(
                name="alpha",
                label="Smoothing (alpha)",
                type="float",
                group=_MODEL,
                default=None,
                min=0.0001,
                max=1_000,
                step=0.1,
                description="Keeps the model from leaning too hard on any one detail: a penalty on large weights for "
                "ridge regression, smoothing of word counts for naive Bayes. Larger is simpler. Leave it unset for 1. "
                "Ridge and naive Bayes only.",
            )
        )

    data: list[ParamSpec] = [seed_spec(_TRAINING)]
    if spec.kind == "classify":
        data.append(
            ParamSpec(
                name="useClassWeights",
                label="Weight classes by inverse frequency",
                type="bool",
                group=_DATA,
                default=False,
                description="Makes mistakes on rarer classes count for more, so the model doesn't simply favour the "
                "most common one. Not used by naive Bayes.",
            )
        )
    if spec.input == "text":
        data += [
            ParamSpec(
                name="analyzer",
                label="Split text into",
                type="choice",
                group=_DATA,
                default="word",
                choices=list(tasks.ANALYZERS),
                description="word: whole words and phrases of words. char_wb: runs of characters inside words, which "
                "copes with typos and languages that do not separate words with spaces.",
            ),
            ParamSpec(
                name="ngramMax",
                label="Longest phrase (words or characters)",
                type="int",
                group=_DATA,
                default=tasks.DEFAULT_NGRAM_MAX,
                min=1,
                max=3,
                step=1,
                description='1 looks at single words, 2 adds pairs of neighbouring words ("not good"), 3 adds '
                "triples. Longer phrases catch more context but make the model much bigger.",
            ),
            ParamSpec(
                name="maxFeatures",
                label="Vocabulary size",
                type="choice",
                group=_DATA,
                default=str(tasks.DEFAULT_MAX_FEATURES),
                choices=[str(n) for n in tasks.MAX_FEATURES_CHOICES],
                description="Keeps only this many of the most frequent words and phrases. Bigger remembers more "
                "rare ones but is slower and can memorise.",
            ),
            ParamSpec(
                name="minDf",
                label="Ignore rare terms seen in fewer than",
                type="int",
                group=_DATA,
                default=1,
                min=1,
                max=1000,
                step=1,
                description="Drops words and phrases that appear in fewer than this many texts, which removes typos "
                "and one-off noise.",
            ),
            ParamSpec(
                name="sublinearTf",
                label="Dampen repeated words",
                type="bool",
                group=_DATA,
                default=True,
                description="Counts a word that appears ten times as a bit more than once, not ten times more. "
                "Usually helps.",
            ),
        ]
    return [*training, *data]  # fmt: skip
