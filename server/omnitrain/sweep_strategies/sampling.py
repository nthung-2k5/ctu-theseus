"""Strategies that draw combinations at random; seed `rng` and the same draw comes back."""

import sys

from omnitrain.sweep_strategies._space import grid_size, grid_trial, sample_random
from omnitrain.sweep_strategies.base import SweepStrategy


class Random(SweepStrategy):
    id = "random"
    label = "Random: sample combinations"
    description = (
        "Picks each setting at random for every trial, which is quicker than a grid when there are many "
        "combinations. The same combination can come up twice."
    )
    order = 20

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        return sample_random(search_space, max_trials, rng)


class RandomUnique(SweepStrategy):
    id = "random_unique"
    label = "Random, no repeats"
    description = (
        "Samples random combinations but never trains the same one twice. When the trial limit covers the whole "
        "grid it runs all of it, in random order, so no run is wasted on a duplicate."
    )
    order = 30

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        total = grid_size(search_space)
        if total <= sys.maxsize:
            picks = rng.sample(range(total), min(max_trials, total))
        else:
            # Too many combinations to index (random.sample needs a length): rejection sampling is fine here
            # because the space is astronomically larger than the trial limit.
            seen: set[int] = set()
            while len(seen) < max_trials:
                seen.add(rng.randrange(total))
            picks = list(seen)
            rng.shuffle(picks)
        return [grid_trial(search_space, index) for index in picks]


class Balanced(SweepStrategy):
    id = "balanced"
    label = "Balanced: even coverage of each setting"
    description = (
        "A random sample that gives every value of every setting about the same number of trials (Latin "
        "hypercube style), so a small budget does not leave a value untried by chance."
    )
    order = 40

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        columns: dict[str, list] = {}
        for key, values in search_space.items():
            order = list(values)
            rng.shuffle(order)
            # Cycle through a shuffled copy, so each value is used floor(n/k) or ceil(n/k) times, then mix the
            # column so the knobs are not paired in a fixed pattern.
            column = [order[i % len(order)] for i in range(max_trials)]
            rng.shuffle(column)
            columns[key] = column
        return [{key: column[i] for key, column in columns.items()} for i in range(max_trials)]
