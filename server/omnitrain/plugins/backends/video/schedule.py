"""The learning-rate curve of a video training run, as a pure function of the step so tests need no optimizer."""

import math


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """The share of the base learning rate to use at optimizer step `step` (0-based) of `total_steps`.

    Linear warmup from near zero over `warmup_steps`, then `schedule` over the remaining steps: "constant" stays at 1,
    "linear" falls to 0 on the last step and "cosine" follows half a cosine wave down to 0.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "linear":
        return 1.0 - progress
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f'Unknown learning rate schedule "{schedule}"')
