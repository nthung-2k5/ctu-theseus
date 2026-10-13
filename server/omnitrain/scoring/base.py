"""The headline score plugin base class.

A run's headline score is the one number the runs list, the "best run" marker and the comparison show
(`services/evaluation_score.py`). It is read off the evaluation report's `overall` block, which each trainer
backend fills with the metrics that fit its output. To make a new kind of metric the headline, create a module
in this package with a subclass that sets `id` (the key in `overall`), `label` and, if lower is better,
`higher_is_better = False`. Restart the backend; nothing else needs an edit, because the web reads the label and
direction off each run summary (`evaluation.score`).

When a report carries several known metrics the one with the lowest `order` wins, so `order` says which is most
telling: accuracy before a regression error, say.
"""

from typing import Any, ClassVar

from omnitrain.core.plugins import Registry

_registry: Registry["HeadlineScore"] = Registry("headline score", "omnitrain.scoring")


class HeadlineScore:
    id: ClassVar[str]
    label: ClassVar[str]
    higher_is_better: ClassVar[bool] = True
    # Priority among the metrics a report carries: lowest wins.
    order: ClassVar[int] = 1000

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "id" in cls.__dict__ and not hasattr(cls, "label"):
            raise TypeError(f"Headline score {cls.id!r} ({cls.__module__}) needs a label")
        _registry.register(cls)

    @classmethod
    def value(cls, overall: dict[str, Any]) -> float | None:
        """This metric's value in a report's `overall` block, or None when the report does not carry it.

        Override for a metric stored under a different shape; the default reads a plain number at `id`.
        """
        raw = overall.get(cls.id)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        return None


def find_score(score_id: str) -> type[HeadlineScore] | None:
    return _registry.find(score_id)


def list_scores() -> list[type[HeadlineScore]]:
    """Installed scores, most telling first."""
    return sorted(_registry.all(), key=lambda s: (s.order, s.id))
