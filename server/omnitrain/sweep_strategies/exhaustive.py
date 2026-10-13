"""Strategies that walk the space in a fixed order: the same space always gives the same trials."""

from omnitrain.sweep_strategies._space import expand_grid
from omnitrain.sweep_strategies.base import SweepStrategy


class Grid(SweepStrategy):
    id = "grid"
    label = "Grid: every combination"
    description = (
        "Tries every possible combination of the values you list, in order. If that is more than the trial "
        "limit, only the first combinations are run, so set the limit to the full count to cover them all."
    )
    order = 10

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        return expand_grid(search_space)[:max_trials]


class OneAtATime(SweepStrategy):
    id = "one_at_a_time"
    label = "One at a time: vary a single setting"
    description = (
        "Trains a baseline (the first value of every setting), then changes one setting at a time while the "
        "rest stay at the baseline. Few runs, and it shows which setting matters most; it cannot find "
        "combinations that only work together."
    )
    order = 50

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        baseline = {key: values[0] for key, values in search_space.items()}
        trials = [baseline]
        for key, values in search_space.items():
            trials.extend({**baseline, key: value} for value in values[1:])
        return trials[:max_trials]
