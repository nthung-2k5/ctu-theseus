# Sweep strategies

A sweep is N ordinary training runs enqueued at once (see `services/sweep.py`). A strategy has one job: given the
candidate values per hyperparameter and a trial budget, say which combinations to train. Five ship today: `grid`,
`random`, `random_unique`, `balanced` and `one_at_a_time`.

`sweeps.strategy` is free text (migration 0016) holding the strategy id, `GET /api/sweep-strategies` lists them and
the web's selector renders that list. `queue_sweep` refuses an unknown id. Adding one needs no enum, migration or
frontend edit.

## How a strategy is found

`Registry` imports every module directly under `omnitrain.sweep_strategies` whose name does not start with `_`
(`_space.py` holds the shared helpers). A `SweepStrategy` subclass registers itself when created, but only if it sets
its **own** `id`. It must also set a `label`, or class creation raises `TypeError`.

## Adding a strategy

1. **Create a module** `sweep_strategies/<name>.py`, or add a class to `exhaustive.py` (fixed order) or
   `sampling.py` (random).
2. **Subclass `SweepStrategy`** and set `id`, `label`, `description` (one or two sentences for the tooltip: what is
   tried and when to pick it) and `order` (selector order; ties fall back to `id`).
3. **Implement `trials(search_space, max_trials, rng)`**, a classmethod. `search_space` maps each hyperparameter key
   to its list of candidate values. Return a list of dicts, each mapping **every** key to **one** of its candidates.
   `services/sweep.expand_sweep` truncates anything longer than `max_trials`, but return no more than you need.
4. **Reuse the helpers** in `_space.py` (`expand_grid`, `grid_trial`, ...) rather than reimplementing them.
5. **Test** it like `server/tests/test_sweep_strategies.py`. That file parametrizes over every registered
   strategy, so a new one is checked automatically for: candidate values only, every key named, never over budget, and
   a seeded `rng` reproducing the same trials. Add a test for what is specific to yours.

## Minimal example

```python
# sweep_strategies/extremes.py
from omnitrain.sweep_strategies.base import SweepStrategy


class Extremes(SweepStrategy):
    id = "extremes"
    label = "Extremes: first and last values"
    description = "Trains the all-first-values trial and the all-last-values trial. Two runs, a quick bracket."
    order = 60

    @classmethod
    def trials(cls, search_space, max_trials, rng):
        first = {key: values[0] for key, values in search_space.items()}
        last = {key: values[-1] for key, values in search_space.items()}
        return [first, last][:max_trials]
```

## Rules that are easy to break

- **Use the `rng` you are given**, never the global `random`, so a seeded sweep reproduces.
- **A strategy cannot see earlier results.** All trials are returned up front and the train lane runs them one
  after another. An adaptive strategy (Bayesian, successive halving) would first need something that enqueues more
  trials as runs finish; that does not exist yet.
- **Do not assume a key's type.** Values can be numbers, strings or booleans, and the model-choice key varies per
  backend (`encoderId` for Ludwig, `modelId` for the rest).
