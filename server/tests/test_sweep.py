import random

from theseus.services.sweep import expand_grid, expand_sweep, sample_random, validate_search_space


class TestValidateSearchSpace:
    def test_rejects_empty_search_space(self):
        assert "at least one hyperparameter" in validate_search_space({}, 5)

    def test_rejects_knob_with_no_candidates(self):
        assert "learningRate" in validate_search_space({"learningRate": []}, 5)

    def test_rejects_max_trials_outside_1_to_50(self):
        assert "maxTrials" in validate_search_space({"learningRate": [0.01]}, 0)
        assert "maxTrials" in validate_search_space({"learningRate": [0.01]}, 51)

    def test_accepts_valid_search_space(self):
        assert validate_search_space({"learningRate": [0.01, 0.001]}, 4) is None
        assert validate_search_space({"learningRate": [0.01]}, 50) is None


class TestExpandGrid:
    def test_full_cartesian_product(self):
        trials = expand_grid({"learningRate": [0.01, 0.001], "batchSize": [16, 32]})
        assert len(trials) == 4
        for expected in (
            {"learningRate": 0.01, "batchSize": 16},
            {"learningRate": 0.01, "batchSize": 32},
            {"learningRate": 0.001, "batchSize": 16},
            {"learningRate": 0.001, "batchSize": 32},
        ):
            assert expected in trials

    def test_first_knob_varies_slowest(self):
        trials = expand_grid({"a": [1, 2], "b": ["x", "y"]})
        assert trials == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]

    def test_single_knob_gives_one_trial_per_candidate(self):
        assert expand_grid({"encoderId": ["resnet18", "resnet50", "vit_base"]}) == [
            {"encoderId": "resnet18"},
            {"encoderId": "resnet50"},
            {"encoderId": "vit_base"},
        ]

    def test_empty_search_space_gives_exactly_one_empty_trial(self):
        assert expand_grid({}) == [{}]


class TestSampleRandom:
    def test_exactly_count_trials_drawn_from_candidates(self):
        space = {"learningRate": [0.01, 0.001], "encoderId": ["resnet18", "resnet50"]}
        trials = sample_random(space, 10)
        assert len(trials) == 10
        for t in trials:
            assert t["learningRate"] in space["learningRate"]
            assert t["encoderId"] in space["encoderId"]

    def test_zero_count_gives_no_trials(self):
        assert sample_random({"learningRate": [0.01]}, 0) == []

    def test_is_deterministic_with_a_seeded_rng(self):
        space = {"batchSize": [16, 32, 64, 128]}
        assert sample_random(space, 8, random.Random(7)) == sample_random(space, 8, random.Random(7))


class TestExpandSweep:
    def test_grid_truncates_to_max_trials_rather_than_sampling(self):
        assert expand_sweep({"batchSize": [16, 32, 64, 128]}, "grid", 2) == [{"batchSize": 16}, {"batchSize": 32}]

    def test_grid_returns_full_product_when_max_trials_exceeds_it(self):
        assert len(expand_sweep({"batchSize": [16, 32]}, "grid", 10)) == 2

    def test_random_always_returns_exactly_max_trials(self):
        assert len(expand_sweep({"batchSize": [16, 32]}, "random", 7)) == 7
