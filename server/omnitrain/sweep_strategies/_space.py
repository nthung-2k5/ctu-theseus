"""Search-space helpers shared by the strategies (underscore-prefixed, so not loaded as a strategy module).

Keys are the camelCase names of the chosen trainer backend's Hyperparameters model (learningRate, batchSize,
modelId, ...), exactly as they arrive in the API and are stored in sweeps.search_space.
"""

import itertools
import math
import random
from typing import Any

SearchSpace = dict[str, list[Any]]


def grid_size(search_space: SearchSpace) -> int:
    """How many combinations the full grid has. An empty space has exactly one (the empty trial)."""
    return math.prod(len(values) for values in search_space.values())


def expand_grid(search_space: SearchSpace) -> list[dict[str, Any]]:
    """Every combination of the candidate values (cartesian product), in a stable order.

    The first key varies slowest. An empty search space yields exactly one empty trial.
    """
    keys = list(search_space)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(search_space[k] for k in keys))]


def grid_trial(search_space: SearchSpace, index: int) -> dict[str, Any]:
    """The `index`-th combination of the grid, in `expand_grid` order, without building the grid."""
    trial: dict[str, Any] = {}
    for key in reversed(list(search_space)):
        index, pick = divmod(index, len(search_space[key]))
        trial[key] = search_space[key][pick]
    return {key: trial[key] for key in search_space}


def sample_random(search_space: SearchSpace, count: int, rng: random.Random | None = None) -> list[dict[str, Any]]:
    """`count` combinations, each knob sampled independently and uniformly from its candidates."""
    rng = rng or random.Random()
    return [{key: rng.choice(values) for key, values in search_space.items()} for _ in range(count)]
