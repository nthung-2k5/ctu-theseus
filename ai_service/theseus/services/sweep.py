"""Hyperparameter sweep search-space expansion.

Deliberately not Ludwig's own hyperopt (Ray Tune-backed): that would contend with training for
the single GPU, its progress is opaque to the run/metric/log event pipeline, and it reuses none
of the cancel and recovery machinery an ordinary run already gets. Instead a sweep is just N
ordinary training runs: this module turns a search space into that list of N selections, which
the sweeps router then enqueues one after another through the normal training path.

Keys are the camelCase names of TrainerSelections (learningRate, batchSize, encoderId, ...),
exactly as they arrive in the API and are stored in sweeps.search_space.
"""

import itertools
import random
from typing import Any, Literal

MAX_TRIALS_CAP = 50

SearchSpace = dict[str, list[Any]]


def validate_search_space(search_space: SearchSpace, max_trials: int) -> str | None:
    """Returns an error message, or None when the search space is valid."""
    if not search_space:
        return "Search space must specify at least one hyperparameter"
    for key, values in search_space.items():
        if not values:
            return f"'{key}' must list at least one candidate value"
    if max_trials < 1 or max_trials > MAX_TRIALS_CAP:
        return f"maxTrials must be between 1 and {MAX_TRIALS_CAP}"
    return None


def expand_grid(search_space: SearchSpace) -> list[dict[str, Any]]:
    """Every combination of the candidate values (cartesian product), in a stable order.

    The first key varies slowest. An empty search space yields exactly one empty trial.
    """
    keys = list(search_space)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(search_space[k] for k in keys))]


def sample_random(search_space: SearchSpace, count: int, rng: random.Random | None = None) -> list[dict[str, Any]]:
    """`count` combinations, each knob sampled independently and uniformly from its candidates."""
    rng = rng or random.Random()
    return [{key: rng.choice(values) for key, values in search_space.items()} for _ in range(count)]


def expand_sweep(
    search_space: SearchSpace, strategy: Literal["grid", "random"], max_trials: int, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """Expand a search space into the trials to dispatch.

    `grid` is capped at max_trials by truncating, not sampling down (a user who wants the full
    grid sets max_trials to the product size), so a large space cannot accidentally enqueue
    hundreds of runs against the single GPU.
    """
    if strategy == "grid":
        return expand_grid(search_space)[:max_trials]
    return sample_random(search_space, max_trials, rng)
