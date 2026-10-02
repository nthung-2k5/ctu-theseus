"""drop inference_jobs

Revision ID: 0007
Revises: 0006

Predictions now run inside the request that asked for them (services/inference.py), so nothing about
an inference is stored: no job rows, no queue lane. Dropping the table also drops its status enum.
The downgrade recreates the empty table; the rows themselves are gone for good.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(op.f("ix_inference_jobs_status"), table_name="inference_jobs")
    op.drop_index(op.f("ix_inference_jobs_run_id"), table_name="inference_jobs")
    op.drop_table("inference_jobs")
    op.execute("DROP TYPE inference_job_status")


def downgrade() -> None:
    op.create_table(
        "inference_jobs",
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("pending", "running", "success", "failed", name="inference_job_status"),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=True),
        sa.Column("upload_key", sa.Text(), nullable=True),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_inference_jobs_run_id"), "inference_jobs", ["run_id"], unique=False)
    op.create_index(op.f("ix_inference_jobs_status"), "inference_jobs", ["status"], unique=False)
