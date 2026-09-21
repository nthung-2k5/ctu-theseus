import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db import enums as e
from theseus.db.base import Base
from theseus.db.models._common import created_at, pg_enum, updated_at, uuid_pk
from theseus.db.models.dataset import split_type_t

sweep_strategy_t = pg_enum(e.SweepStrategy, "sweep_strategy")
sweep_status_t = pg_enum(e.SweepStatus, "sweep_status")
training_status_t = pg_enum(e.TrainingStatus, "training_status")
evaluation_split_t = pg_enum(e.EvaluationSplit, "evaluation_split")
evaluation_status_t = pg_enum(e.EvaluationStatus, "evaluation_status")
run_event_kind_t = pg_enum(e.RunEventKind, "run_event_kind")


class JobColumns:
    """Columns that make a domain row claimable as a job (see jobs/queue.py). The status column is the lock."""

    attempt: Mapped[int] = mapped_column(sa.Integer, server_default="0")
    max_attempts: Mapped[int] = mapped_column(sa.Integer, server_default="1")
    available_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())
    claimed_by: Mapped[str | None] = mapped_column(sa.Text)
    lease_expires_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(sa.Text)


class Sweep(Base):
    __tablename__ = "sweeps"

    id: Mapped[uuid.UUID] = uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("dataset_versions.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(sa.String(255))
    # Maps each trainer knob to its candidate values, see services/sweep.py.
    search_space: Mapped[Any] = mapped_column(JSONB)
    strategy: Mapped[str] = mapped_column(sweep_strategy_t)
    max_trials: Mapped[int] = mapped_column(sa.Integer)
    status: Mapped[str] = mapped_column(sweep_status_t, server_default="running")
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()


class TrainingRun(JobColumns, Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[str] = mapped_column(training_status_t, server_default="queued", index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("dataset_versions.id", ondelete="CASCADE"))
    # Set only for a run dispatched as one trial of a sweep.
    sweep_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("sweeps.id", ondelete="CASCADE"), index=True)
    trial_index: Mapped[int | None] = mapped_column(sa.Integer)
    hyperparameters: Mapped[Any] = mapped_column(JSONB)
    # The exact compiled Ludwig config, for reproducibility.
    ludwig_config: Mapped[Any | None] = mapped_column(JSONB)
    config_key: Mapped[str | None] = mapped_column(sa.Text)
    best_epoch: Mapped[int | None] = mapped_column(sa.Integer)
    # Bumped by the event writer; only used to catch a hung (not crashed) training thread.
    heartbeat_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    # Durable cancel intent; the in-memory AbortRegistry is only the fast read path.
    cancel_requested_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    failed_message: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    training_run_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    epoch: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    split: Mapped[str] = mapped_column(split_type_t, primary_key=True)
    metric_name: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    metric_value: Mapped[float] = mapped_column(sa.REAL)
    created_at: Mapped[datetime] = created_at()


class RunEvaluation(Base):
    """One row per run, so run_id is both the natural key and the primary key."""

    __tablename__ = "run_evaluations"

    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(evaluation_status_t)
    split: Mapped[str | None] = mapped_column(evaluation_split_t)
    report_key: Mapped[str | None] = mapped_column(sa.Text)
    predictions_key: Mapped[str | None] = mapped_column(sa.Text)
    report: Mapped[Any | None] = mapped_column(JSONB)
    # Denormalized from report for cheap sorting/filtering without parsing jsonb.
    accuracy: Mapped[float | None] = mapped_column(sa.REAL)
    macro_f1: Mapped[float | None] = mapped_column(sa.REAL)
    failed_message: Mapped[str | None] = mapped_column(sa.Text)
    evaluated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())


class RunEvent(Base):
    """Replayable event log behind the SSE endpoint. seq is the SSE event id.

    Only events.writer inserts here. A single writer keeps seq order equal to commit order,
    which Last-Event-ID replay depends on.
    """

    __tablename__ = "run_events"
    __table_args__ = (sa.Index("run_events_run_id_seq_idx", "run_id", "seq"),)

    seq: Mapped[int] = mapped_column(sa.BigInteger, sa.Identity(), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("training_runs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(run_event_kind_t)
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())
    payload: Mapped[Any] = mapped_column(JSONB)
