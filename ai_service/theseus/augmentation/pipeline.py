"""Run a configured set of augmentation ops over samples, deterministically."""

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from theseus.augmentation.base import Augmentation, ParamsModel
from theseus.augmentation.config import AugmentationConfig
from theseus.augmentation.registry import get_augmentation


@dataclass
class ConfiguredOp:
    op: type[Augmentation]
    params: ParamsModel
    probability: float
    state: Any = None

    def record(self) -> dict[str, Any]:
        """What was applied, stored on the augmented item so the UI can show it."""
        return {"id": self.op.id, "params": self.params.model_dump(by_alias=True)}


def build_plan(config: AugmentationConfig, originals: Sequence[Any]) -> list[ConfiguredOp]:
    """Resolve a validated config into ops with parsed params and their dataset-level state."""
    plan = []
    for op_config in config.ops:
        op = get_augmentation(op_config.id)
        plan.append(
            ConfiguredOp(
                op=op,
                params=op.Params.model_validate(op_config.params),
                probability=op_config.probability,
                state=op.prepare(originals),
            )
        )
    return plan


def seed_for(version_id: object, item_id: object, copy_index: int) -> int:
    """Stable per (snapshot, item, copy): rebuilding a snapshot reproduces the same augmented items."""
    digest = hashlib.sha256(f"{version_id}:{item_id}:{copy_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def augment(sample: Any, plan: Sequence[ConfiguredOp], rng: random.Random) -> tuple[Any, list[dict[str, Any]]]:
    """Apply each op with its probability. If none fires, one is forced so a copy is never a plain duplicate."""
    chosen = [c for c in plan if rng.random() < c.probability]
    if not chosen:
        chosen = [rng.choice(plan)]
    for c in chosen:
        sample = c.op.apply(sample, c.params, rng, c.state)
    return sample, [c.record() for c in chosen]
