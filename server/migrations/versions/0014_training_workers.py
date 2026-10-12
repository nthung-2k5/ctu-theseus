"""training workers: `training_workers` table and `training_runs.worker_id`

Revision ID: 0014
Revises: 0013

A worker is a remote GPU machine that pulls training runs over HTTP. `worker_id` is NULL for a run that trains on
the API server itself (every existing run), and `ON DELETE SET NULL` so removing a worker row never removes a run.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_workers",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("uuidv7()")),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("token_hash", sa.CHAR(64), nullable=False, unique=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("info", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "training_runs",
        sa.Column("worker_id", sa.Uuid(), sa.ForeignKey("training_workers.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index("ix_training_runs_worker_id", "training_runs", ["worker_id"])


def downgrade() -> None:
    op.drop_index("ix_training_runs_worker_id", table_name="training_runs")
    op.drop_column("training_runs", "worker_id")
    op.drop_table("training_workers")
