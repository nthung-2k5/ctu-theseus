"""The wire contract between the API and a remote training worker (`omnitrain.worker`)."""

import uuid
from typing import Any, Literal

from pydantic import Field

from omnitrain.db.enums import EvaluationSplit, EvaluationStatus, SplitType
from omnitrain.schemas.common import ApiModel

# Bumped whenever a server and a worker built before the change could no longer work together: the shape of these
# models, the S3 layout of a run's inputs and results, or what `jobs.run_core` expects of a backend. A claim from a
# worker on another protocol is refused (routers/worker.py), because training with mismatched code silently produces
# models the server cannot serve.
#   1  claim / events / complete
#   2  the run spec can carry a bring-your-own model for the worker to fetch (`WorkerCustomModel`)
PROTOCOL_VERSION = 2


class WorkerClaimBody(ApiModel):
    # Trainer backend ids this worker can train right now (installed and `available()`).
    backends: list[str] = Field(max_length=64)
    # `PROTOCOL_VERSION` of the worker's code; the server refuses a claim from a different one.
    protocol: int
    # The worker's package version, for the admin page and error messages only.
    version: str = Field(max_length=64)
    hostname: str | None = Field(default=None, max_length=255)
    gpu: str | None = Field(default=None, max_length=255)


class WorkerCustomModel(ApiModel):
    """A bring-your-own model the worker fetches for itself. The server only hands one out when no secret is needed
    to get it: an uploaded bundle (read from S3) or a public Hub model at a pinned revision."""

    id: uuid.UUID
    kind: str
    source_kind: str
    source_ref: str | None = None
    revision: str | None = None
    sha256: str | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    # Where the compiled config says the model's files are on the SERVER. The worker keeps them at its own path and
    # swaps this prefix for that one wherever it appears in the config.
    server_path: str


class WorkerRunSpec(ApiModel):
    """What a worker needs to train a run: the same fields `jobs.run_core.RunSpec` takes."""

    id: uuid.UUID
    backend: str
    config: dict[str, Any]
    dataset_version_id: uuid.UUID
    config_key: str | None
    custom_model: WorkerCustomModel | None = None


class WorkerMetric(ApiModel):
    epoch: int
    split: SplitType
    metrics: dict[str, float]


class WorkerEventsBody(ApiModel):
    metrics: list[WorkerMetric] = Field(default_factory=list, max_length=500)
    # True when the training thread proved it is alive since the last call (a backend called `heartbeat()`).
    heartbeat: bool = False


class WorkerEventsResponse(ApiModel):
    # The run was canceled: stop at the next safe point.
    cancel: bool
    # The run is already over on the server (canceled, failed by the reaper, ...): stop, and do not report back.
    terminal: bool


class WorkerEvaluation(ApiModel):
    """The `run_evaluations` columns `jobs.run_core.evaluate` produced. The object keys are not taken from the worker:
    the server derives them from the run id."""

    status: EvaluationStatus
    split: EvaluationSplit | None = None
    report: dict[str, Any] | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    failed_message: str | None = Field(default=None, max_length=2000)


class WorkerCompleteBody(ApiModel):
    status: Literal["succeeded", "failed", "canceled"]
    message: str | None = Field(default=None, max_length=4000)
    evaluation: WorkerEvaluation | None = None


class WorkerCompleteResponse(ApiModel):
    # The run's final status on the server (it may differ from what the worker reported if the run had already ended).
    status: str
