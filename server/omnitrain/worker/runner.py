"""The worker's main loop: claim a run, train it, report how it went."""

import functools
import json
import logging
import socket
import threading
import time
import uuid
from typing import Any

from omnitrain import __version__
from omnitrain.jobs import run_core
from omnitrain.jobs.abort import TrainingAborted
from omnitrain.plugins.backends.registry import list_backends
from omnitrain.schemas.worker import PROTOCOL_VERSION
from omnitrain.worker import custom
from omnitrain.worker.client import ApiClient, ApiError, Unauthorized, is_transient
from omnitrain.worker.config import WorkerConfig
from omnitrain.worker.sink import HttpSink

logger = logging.getLogger("omnitrain.worker")

# How long a finished run's outcome is retried before the worker gives up and leaves it to the server's heartbeat
# reaper. A result that never arrives fails the run after `run_heartbeat_timeout_seconds`, so this is bounded anyway.
COMPLETE_ATTEMPTS = 12
COMPLETE_MAX_BACKOFF_SECONDS = 60.0
# How long to wait after a claim the server refused for a reason that will not fix itself in seconds (a version
# mismatch), so the log is not flooded.
REFUSED_BACKOFF_SECONDS = 30.0


@functools.cache
def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def describe() -> dict[str, Any]:
    """The `/claim` body: what this worker is and which backends it can train right now."""
    return {
        "backends": [b.id for b in list_backends() if b.available() is None],
        "protocol": PROTOCOL_VERSION,
        "version": __version__,
        "hostname": socket.gethostname(),
        "gpu": _gpu_name(),
    }


def _jsonable(value: Any) -> Any:
    """Plain JSON types only: an evaluation report can carry numpy scalars, which `json` cannot encode."""

    def default(o: Any) -> Any:
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    return json.loads(json.dumps(value, default=default))


def evaluation_body(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    """`run_core.evaluate`'s columns as the API's camelCase `WorkerEvaluation`. Object keys are left out: the API
    derives them from the run id."""
    if evaluation is None:
        return None
    return _jsonable(
        {
            "status": evaluation["status"],
            "split": evaluation.get("split"),
            "report": evaluation.get("report"),
            "accuracy": evaluation.get("accuracy"),
            "macroF1": evaluation.get("macro_f1"),
            "failedMessage": evaluation.get("failed_message"),
        }
    )


def _spec(raw: dict[str, Any]) -> run_core.RunSpec:
    return run_core.RunSpec(
        id=uuid.UUID(raw["id"]),
        backend=raw["backend"],
        config=raw["config"],
        dataset_version_id=uuid.UUID(raw["datasetVersionId"]),
        config_key=raw.get("configKey"),
    )


def _complete(client: ApiClient, run_id: str, body: dict[str, Any]) -> None:
    delay = 2.0
    for attempt in range(1, COMPLETE_ATTEMPTS + 1):
        try:
            answer = client.complete(run_id, body)
            logger.info("Run %s reported %s (server status: %s)", run_id, body["status"], answer.get("status"))
            return
        except Exception as e:
            if not is_transient(e):
                logger.error("The API refused the outcome of run %s: %s", run_id, e)
                return
            logger.warning("Could not report the outcome of run %s (attempt %d): %s", run_id, attempt, e)
            time.sleep(delay)  # also while shutting down: the outcome matters more than a quick exit
            delay = min(delay * 2, COMPLETE_MAX_BACKOFF_SECONDS)
    logger.error("Gave up reporting run %s; the server will fail it when its heartbeat times out", run_id)


def run_one(client: ApiClient, raw: dict[str, Any], shutdown: threading.Event, flush_interval: float) -> None:
    """Train one claimed run and report its outcome. Never raises for a training failure: that is an outcome."""
    spec = _spec(raw)
    run_id = str(spec.id)
    logger.info("Training run %s with the %s backend", run_id, spec.backend)
    sink = HttpSink(client, run_id, shutdown, flush_interval)
    sink.start()
    status, message, evaluation = "succeeded", None, None
    try:
        if raw.get("customModel"):
            logger.info("Fetching the base model for run %s", run_id)
            spec.config = custom.prepare(raw["customModel"], spec.config)
        evaluation = run_core.execute(spec, sink)
    except TrainingAborted:
        if shutdown.is_set():
            # Not a cancel by anyone: the machine is going away. Failing it lets the owner see why and train again.
            status, message = "failed", "The worker was shut down during training"
        else:
            status = "canceled"
    except Exception as e:
        logger.exception("Training failed for run %s", run_id)
        status, message = "failed", f"{type(e).__name__}: {e}"
    finally:
        sink.stop()

    if sink.terminal:
        logger.info("Run %s was already ended by the server; nothing to report", run_id)
        return
    body = {"status": status, "message": message, "evaluation": evaluation_body(evaluation)}
    _complete(client, run_id, body)


def serve(config: WorkerConfig, client: ApiClient, shutdown: threading.Event, *, max_runs: int | None = None) -> int:
    """Claim and train runs until `shutdown` is set (or `max_runs` have been trained). Returns an exit code."""
    done = 0
    while not shutdown.is_set() and (max_runs is None or done < max_runs):
        try:
            raw = client.claim(describe())
        except Unauthorized as e:
            logger.error("The server rejected this worker's token (%s). Register the worker again.", e.message)
            return 1
        except ApiError as e:
            if e.retryable:
                logger.warning("The server could not answer a claim: %s", e)
                shutdown.wait(config.poll_interval_seconds)
            else:
                logger.error("The server refused this worker: %s", e.message)
                shutdown.wait(REFUSED_BACKOFF_SECONDS)
            continue
        except Exception as e:
            logger.warning("Could not reach the server at %s: %s", config.api_url, e)
            shutdown.wait(config.poll_interval_seconds)
            continue
        if raw is None:
            shutdown.wait(config.poll_interval_seconds)
            continue
        run_one(client, raw, shutdown, config.flush_interval_seconds)
        done += 1
    return 0
