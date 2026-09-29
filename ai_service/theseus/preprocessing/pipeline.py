"""Resolve a configured set of preprocessing ops over samples. No rng: same input, same output."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from theseus.preprocessing.base import ParamsModel, Preprocessing
from theseus.preprocessing.config import PreprocessingConfig, SplitName
from theseus.preprocessing.registry import get_preprocessing


@dataclass
class ConfiguredOp:
    op: type[Preprocessing]
    params: ParamsModel
    splits: set[SplitName]
    state: Any = None

    def record(self) -> dict[str, Any]:
        """What was applied, stored on the preprocessed item so the UI can show it."""
        return {"id": self.op.id, "params": self.params.model_dump(by_alias=True)}


def build_plan(config: PreprocessingConfig, train_samples: Sequence[Any]) -> list[ConfiguredOp]:
    """Resolve a validated config into ops with parsed params and their train-fitted state."""
    plan = []
    for op_config in config.ops:
        op = get_preprocessing(op_config.id)
        plan.append(
            ConfiguredOp(
                op=op,
                params=op.Params.model_validate(op_config.params),
                splits=set(op_config.splits),
                state=op.fit(train_samples),
            )
        )
    return plan


def ops_for_split(plan: list[ConfiguredOp], split: SplitName) -> list[ConfiguredOp]:
    """The ops (in configured order) that apply to a given item's split."""
    return [c for c in plan if split in c.splits]


def preprocess(sample: Any, ops: Sequence[ConfiguredOp]) -> tuple[Any, list[dict[str, Any]]]:
    """Apply every op selected for this item's split, in order."""
    applied = []
    for c in ops:
        sample = c.op.apply(sample, c.params, c.state)
        applied.append(c.record())
    return sample, applied
