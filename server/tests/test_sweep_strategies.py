"""How a search space becomes trials is a plugin: `omnitrain/sweep_strategies/`."""

import itertools
import random
from collections import Counter

import pytest

from omnitrain.services.sweep import expand_grid, expand_sweep
from omnitrain.sweep_strategies import SweepStrategy, find_strategy, get_strategy, list_strategies
from omnitrain.sweep_strategies._space import grid_trial
from omnitrain.sweep_strategies.base import _registry

SPACE = {"learningRate": [0.01, 0.001, 0.0001], "batchSize": [16, 32], "encoderId": ["a", "b", "c", "d"]}


def trials(strategy: str, space=SPACE, n=10, seed=1):
    return expand_sweep(space, strategy, n, random.Random(seed))


def test_the_builtin_strategies_are_listed_in_display_order():
    assert [s.id for s in list_strategies()] == ["grid", "random", "random_unique", "balanced", "one_at_a_time"]
    assert all(s.label and s.description for s in list_strategies())
    assert find_strategy("bayesian") is None
    with pytest.raises(KeyError):
        get_strategy("bayesian")


@pytest.mark.parametrize("strategy", [s.id for s in list_strategies()])
class TestEveryStrategy:
    def test_trials_only_use_candidate_values_and_name_every_knob(self, strategy):
        for trial in trials(strategy):
            assert set(trial) == set(SPACE)
            assert all(trial[key] in values for key, values in SPACE.items())

    def test_never_more_than_the_budget(self, strategy):
        for budget in (1, 3, 10, 50):
            assert 1 <= len(trials(strategy, n=budget)) <= budget

    def test_a_seeded_rng_reproduces_the_trials(self, strategy):
        assert trials(strategy, seed=5) == trials(strategy, seed=5)


class TestGridTrial:
    def test_indexes_the_grid_in_expand_grid_order(self):
        grid = expand_grid(SPACE)
        assert [grid_trial(SPACE, i) for i in range(len(grid))] == grid


class TestOneAtATime:
    def test_starts_from_a_baseline_then_varies_one_knob_at_a_time(self):
        space = {"lr": [1, 2, 3], "bs": ["x", "y"]}
        assert trials("one_at_a_time", space, 50) == [
            {"lr": 1, "bs": "x"},
            {"lr": 2, "bs": "x"},
            {"lr": 3, "bs": "x"},
            {"lr": 1, "bs": "y"},
        ]

    def test_every_trial_differs_from_the_baseline_in_at_most_one_knob(self):
        baseline = {k: v[0] for k, v in SPACE.items()}
        for trial in trials("one_at_a_time", n=50):
            assert sum(trial[k] != baseline[k] for k in SPACE) <= 1

    def test_the_budget_truncates_it_and_the_baseline_always_comes_first(self):
        assert trials("one_at_a_time", {"lr": [1, 2, 3], "bs": ["x", "y"]}, 2) == [
            {"lr": 1, "bs": "x"},
            {"lr": 2, "bs": "x"},
        ]

    def test_uses_far_fewer_runs_than_the_grid(self):
        assert len(trials("one_at_a_time", n=50)) == 1 + 2 + 1 + 3 < len(expand_grid(SPACE))


class TestRandomUnique:
    def test_never_repeats_a_combination(self):
        out = trials("random_unique", n=20)
        assert len(out) == 20 and len({tuple(t.items()) for t in out}) == 20

    def test_a_budget_larger_than_the_grid_runs_the_whole_grid_once(self):
        space = {"a": [1, 2], "b": ["x", "y", "z"]}
        out = trials("random_unique", space, 50)
        assert sorted(map(tuple, (t.items() for t in out))) == sorted(
            map(tuple, (t.items() for t in expand_grid(space)))
        )

    def test_an_astronomically_large_space_is_sampled_without_building_it(self):
        space = {f"k{i}": list(range(100)) for i in range(12)}  # 100**12 combinations
        out = trials("random_unique", space, 5)
        assert len(out) == 5 and len({tuple(t.items()) for t in out}) == 5


class TestBalanced:
    def test_every_value_of_every_knob_is_used_about_equally(self):
        out = trials("balanced", n=12)
        for key, values in SPACE.items():
            counts = Counter(t[key] for t in out)
            assert set(counts) == set(values)
            assert max(counts.values()) - min(counts.values()) <= 1

    def test_a_small_budget_still_tries_every_value_that_fits(self):
        out = trials("balanced", {"encoderId": ["a", "b", "c", "d"]}, 4)
        assert sorted(t["encoderId"] for t in out) == ["a", "b", "c", "d"]


def test_grid_and_random_keep_their_original_behaviour():
    assert trials("grid", {"batchSize": [16, 32, 64, 128]}, 2) == [{"batchSize": 16}, {"batchSize": 32}]
    assert len(trials("random", {"batchSize": [16, 32]}, 7)) == 7  # with replacement, so repeats are expected


class TestPlugins:
    @pytest.fixture
    def strategies(self):
        _registry.ensure_loaded()
        before = set(_registry._items)
        yield _registry
        for plugin_id in set(_registry._items) - before:
            _registry.unregister(plugin_id)

    def test_a_new_class_is_used_with_no_other_edit(self, strategies):
        class Corners(SweepStrategy):
            id = "corners"
            label = "Corners"
            order = 5

            @classmethod
            def trials(cls, search_space, max_trials, rng):
                lows = {k: v[0] for k, v in search_space.items()}
                highs = {k: v[-1] for k, v in search_space.items()}
                return [lows, highs]

        assert [s.id for s in list_strategies()][0] == "corners"
        assert trials("corners", n=10) == [
            {"learningRate": 0.01, "batchSize": 16, "encoderId": "a"},
            {"learningRate": 0.0001, "batchSize": 32, "encoderId": "d"},
        ]

    def test_a_strategy_that_returns_too_many_is_cut_to_the_budget(self, strategies):
        class Greedy(SweepStrategy):
            id = "greedy"
            label = "Greedy"

            @classmethod
            def trials(cls, search_space, max_trials, rng):
                return list(itertools.islice(itertools.cycle(expand_grid(search_space)), 100))

        assert len(trials("greedy", n=7)) == 7

    def test_a_strategy_without_a_label_or_with_a_taken_id_is_refused(self, strategies):
        with pytest.raises(TypeError, match="needs a label"):

            class Unlabelled(SweepStrategy):
                id = "unlabelled"

        with pytest.raises(ValueError, match="Duplicate sweep strategy id"):

            class Again(SweepStrategy):
                id = "grid"
                label = "Another grid"
