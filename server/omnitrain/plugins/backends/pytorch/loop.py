"""The pieces of a hand-written training loop: heartbeats, the optimizer step, what is being optimised, and `fit`.

Task-neutral: a pass needs a loader that yields `(inputs, targets)`, a network that maps inputs to outputs and an
`Objective` that says how an output is scored against a target. Nothing here knows about images or tables. Imports torch
(lazily imported module).
"""

import contextlib
import time
from collections.abc import Callable
from typing import Any

import torch
from opentelemetry import trace
from torch.utils.data import DataLoader

from omnitrain.plugins.backends.base import TrainContext
from omnitrain.plugins.backends.schedule import lr_factor

tracer = trace.get_tracer("omnitrain")

# The run is presumed hung when it goes `run_heartbeat_timeout_seconds` (15 min) without a heartbeat, which a long
# epoch can exceed. Beat from the batch loop too, this often.
HEARTBEAT_EVERY_SECONDS = 10.0


class Beat:
    """Heartbeat at most every HEARTBEAT_EVERY_SECONDS, so calling it per batch stays cheap."""

    def __init__(self, run: TrainContext) -> None:
        self._run = run
        self._last = 0.0

    def __call__(self) -> None:
        now = time.monotonic()
        if now - self._last >= HEARTBEAT_EVERY_SECONDS:
            self._last = now
            self._run.heartbeat()


class Stepper:
    """Everything one optimizer step needs beyond the optimizer: the learning-rate curve, gradient clipping and mixed
    precision (a loss scaler and `autocast`, both only on a CUDA device, where half precision is worth it)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: list[torch.nn.Parameter],
        device: str,
        hp: dict[str, Any],
        steps_per_epoch: int,
    ) -> None:
        self.optimizer = optimizer
        self._parameters = parameters
        self._clip = hp.get("gradClipNorm")
        self.amp = bool(hp.get("mixedPrecision")) and device.startswith("cuda")
        self._device_type = "cuda" if device.startswith("cuda") else "cpu"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        total = hp["epochs"] * steps_per_epoch
        warmup = round(float(hp.get("warmupEpochs") or 0) * steps_per_epoch)
        schedule = hp.get("lrSchedule") or "constant"
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total, warmup, schedule)
        )

    def autocast(self) -> Any:
        return torch.autocast(self._device_type, dtype=torch.float16, enabled=self.amp)

    def step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(loss).backward()
        if self._clip is not None:
            # Gradients are still multiplied by the loss scale; undo that first so the norm is the real one.
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._parameters, self._clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()
        self._scheduler.step()


def make_optimizer(name: str, parameters: list[torch.nn.Parameter], lr: float, weight_decay: float | None) -> Any:
    """Each optimizer keeps its own weight decay unless one was asked for: SGD 1e-4, and AdamW torch's 0.01."""
    if name == "SGD":
        decay = 1e-4 if weight_decay is None else weight_decay
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=decay)
    if weight_decay is None:
        return torch.optim.AdamW(parameters, lr=lr)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


class Objective:
    """What a run optimises and reports. `loss` is what is minimised; `sums` are the per-batch totals a pass adds up,
    and `summarise` turns the totals into the metrics reported each epoch. The reported `loss` is always the one
    `events/writer.py` compares across epochs to find the best, so it must not depend on how the run was tuned."""

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sums(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError

    def summarise(self, totals: dict[str, float], seen: int) -> dict[str, float]:
        raise NotImplementedError


class Classification(Objective):
    """Cross entropy over class logits. Optional per-class weights steer the optimisation only: the reported loss is
    always the plain cross entropy, so runs with and without them can be compared."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        self._weights = class_weights

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = None if self._weights is None else self._weights.to(outputs.device)
        return torch.nn.functional.cross_entropy(outputs.float(), targets, weight=weights)

    def sums(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        return {
            "loss": float(torch.nn.functional.cross_entropy(outputs.float(), targets, reduction="sum").item()),
            "correct": float((outputs.argmax(dim=1) == targets).sum().item()),
        }

    def summarise(self, totals: dict[str, float], seen: int) -> dict[str, float]:
        return {"loss": totals["loss"] / max(seen, 1), "accuracy": totals["correct"] / max(seen, 1)}


class Tagging(Objective):
    """Cross entropy over the tag logits of every word `[batch, tokens, tags]`; a position labelled `IGNORE` (padding,
    or a word the model does not read) counts for nothing. A pass counts words, not rows, so the epoch's metrics are per
    word (`run_epoch` only counts rows: `summarise` divides by its own count). Optional per-tag weights steer the
    optimisation only, as in `Classification`, since `O` outnumbers every other tag."""

    IGNORE = -100

    def __init__(self, tag_weights: torch.Tensor | None = None) -> None:
        self._weights = tag_weights

    def _flat(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return outputs.float().reshape(-1, outputs.shape[-1]), targets.reshape(-1)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, flat = self._flat(outputs, targets)
        weights = None if self._weights is None else self._weights.to(outputs.device)
        # `sum` over `max(words, 1)`, not torch's mean: a batch whose words are all ignored would be 0 / 0.
        total = torch.nn.functional.cross_entropy(
            logits, flat, weight=weights, ignore_index=self.IGNORE, reduction="sum"
        )
        counted = flat != self.IGNORE
        scale = weights[flat[counted]].sum() if weights is not None else counted.sum()
        return total / scale.clamp(min=1)

    def sums(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        logits, flat = self._flat(outputs, targets)
        counted = flat != self.IGNORE
        loss = torch.nn.functional.cross_entropy(logits, flat, ignore_index=self.IGNORE, reduction="sum")
        return {
            "loss": float(loss.item()),
            "correct": float(((logits.argmax(dim=1) == flat) & counted).sum().item()),
            "count": float(counted.sum().item()),
        }

    def summarise(self, totals: dict[str, float], seen: int) -> dict[str, float]:
        count = max(totals.get("count", 0.0), 1.0)
        return {"loss": totals["loss"] / count, "accuracy": totals["correct"] / count}


class Regression(Objective):
    """Mean squared error of a number. The network outputs the target in its own units; the error is scaled by the
    target's spread so a column measured in millions trains as well as one measured in fractions. The reported loss
    is the squared error in the target's own units, the same as the scikit-learn backend's."""

    def __init__(self, target_std: float) -> None:
        self._std = target_std

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(((outputs.float().reshape(-1) - targets) / self._std) ** 2)

    def sums(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        error = outputs.float().reshape(-1) - targets
        return {"loss": float((error**2).sum().item()), "abs": float(error.abs().sum().item())}

    def summarise(self, totals: dict[str, float], seen: int) -> dict[str, float]:
        return {"loss": totals["loss"] / max(seen, 1), "mean_absolute_error": totals["abs"] / max(seen, 1)}


def run_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    device: str,
    run: TrainContext,
    beat: Beat,
    stepper: Stepper | None,
    objective: Objective,
) -> dict[str, float]:
    """One pass over `loader`: trains when a stepper is given, otherwise only scores. Returns the epoch's metrics."""
    training = stepper is not None
    net.train(training)
    totals: dict[str, float] = {}
    seen = 0
    for inputs, targets in loader:
        run.check_abort()
        beat()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            with stepper.autocast() if stepper is not None else contextlib.nullcontext():
                outputs = net(inputs)
                loss = objective.loss(outputs, targets)
        if stepper is not None:
            stepper.step(loss)
        for key, value in objective.sums(outputs.detach(), targets).items():
            totals[key] = totals.get(key, 0.0) + value
        seen += len(targets)
    return objective.summarise(totals, seen)


def fit(
    run: TrainContext,
    net: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    hp: dict[str, Any],
    device: str,
    beat: Beat,
    objective: Objective,
    span: str,
    on_epoch: Callable[[int], None] | None = None,
) -> None:
    """Train `net` for up to `hp["epochs"]`, reporting each epoch, and leave it holding the weights of the epoch with
    the lowest validation loss. Stops early after `hp["patience"]` epochs without improvement."""
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = make_optimizer(hp["optimizer"], trainable, hp["learningRate"], hp.get("weightDecay"))
    stepper = Stepper(optimizer, trainable, device, hp, len(train_loader))

    best_loss, best_state, stale = float("inf"), None, 0
    with tracer.start_as_current_span(span):
        for epoch in range(1, hp["epochs"] + 1):
            run.check_abort()
            run.heartbeat()
            if on_epoch is not None:
                on_epoch(epoch)
            run.report(epoch, "train", run_epoch(net, train_loader, device, run, beat, stepper, objective))
            validation = run_epoch(net, val_loader, device, run, beat, None, objective)
            run.report(epoch, "validation", validation)
            if validation["loss"] < best_loss - 1e-6:
                best_loss, stale = validation["loss"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= hp["patience"]:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
