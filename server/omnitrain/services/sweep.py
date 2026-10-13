"""Hyperparameter sweep search-space validation and expansion.

Deliberately not Ludwig's own hyperopt (Ray Tune-backed): that would contend with training for
the single GPU, its progress is opaque to the run/metric/log event pipeline, and it reuses none
of the cancel and recovery machinery an ordinary run already gets. Instead a sweep is just N
ordinary training runs: a sweep strategy plugin (`omnitrain/sweep_strategies/`) turns a search space into that
list of N selections, which the sweeps router then enqueues one after another through the normal training path.

Keys are the camelCase names of the chosen trainer backend's Hyperparameters model (learningRate,
batchSize, encoderId, ...), exactly as they arrive in the API and are stored in sweeps.search_space.
"""

import random
from typing import Any

from omnitrain.sweep_strategies import get_strategy
from omnitrain.sweep_strategies._space import SearchSpace, expand_grid, sample_random

__all__ = ["MAX_TRIALS_CAP", "SearchSpace", "expand_grid", "expand_sweep", "sample_random", "validate_search_space"]

MAX_TRIALS_CAP = 50


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


def expand_sweep(
    search_space: SearchSpace, strategy: str, max_trials: int, rng: random.Random | None = None
) -> list[dict[str, Any]]:
    """Expand a search space into the trials to dispatch, using the installed strategy named `strategy`.

    Never more than `max_trials`, whatever the strategy returns, so a large space cannot accidentally enqueue
    hundreds of runs against the single GPU. Raises KeyError for an id no installed strategy defines.
    """
    return get_strategy(strategy).trials(search_space, max_trials, rng or random.Random())[:max_trials]
