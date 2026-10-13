"""The sweep strategy plugin base class: how a search space becomes the list of trials to train.

A sweep is just N ordinary training runs (see `services/sweep.py` for why it is not Ludwig hyperopt), so a
strategy has one job: given the candidate values per hyperparameter and the trial budget, say which
combinations to try. To add one, create a module in this package with a subclass that sets `id`, `label` and
`description` and implements `trials`. Restart the backend; `GET /api/sweep-strategies` lists it and the web's
strategy selector renders that list, with no enum, migration or frontend edit (`sweeps.strategy` is free text).

A strategy sees the whole space up front and returns every trial at once; the train lane then runs them one
after another. It cannot look at earlier trials' results (nothing adaptive yet).
"""

import random
from typing import Any, ClassVar

from omnitrain.core.plugins import Registry
from omnitrain.sweep_strategies._space import SearchSpace

_registry: Registry["SweepStrategy"] = Registry("sweep strategy", "omnitrain.sweep_strategies")


class SweepStrategy:
    id: ClassVar[str]
    label: ClassVar[str]
    # One or two sentences for the selector's tooltip: what is tried, and when to pick it.
    description: ClassVar[str] = ""
    # Sort key in the selector; ties fall back to id.
    order: ClassVar[int] = 1000

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "id" in cls.__dict__ and not hasattr(cls, "label"):
            raise TypeError(f"Sweep strategy {cls.id!r} ({cls.__module__}) needs a label")
        _registry.register(cls)

    @classmethod
    def trials(cls, search_space: SearchSpace, max_trials: int, rng: random.Random) -> list[dict[str, Any]]:
        """The combinations to train, at most `max_trials` of them (the engine truncates a longer list).

        Each trial maps every key of `search_space` to one of its candidate values. Use `rng`, never the global
        `random`, so a seeded run reproduces."""
        raise NotImplementedError


def get_strategy(strategy_id: str) -> type[SweepStrategy]:
    """The strategy class for an id. Raises KeyError('Unknown sweep strategy ...') if none is installed."""
    return _registry.get(strategy_id)


def find_strategy(strategy_id: str) -> type[SweepStrategy] | None:
    return _registry.find(strategy_id)


def list_strategies() -> list[type[SweepStrategy]]:
    return sorted(_registry.all(), key=lambda s: (s.order, s.id))
