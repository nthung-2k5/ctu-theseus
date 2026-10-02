"""admin area and bring-your-own-model

Revision ID: 0009
Revises: 0008

The admin area and bring-your-own models:

  * users.disabled_at        an admin can disable an account
  * plugin_settings          admin enable/disable switches for plugins, built-in models and tasks; no row
                             means enabled, so an existing install behaves exactly as before
  * custom_models            bring-your-own models (a Hub reference or an uploaded bundle), claimed as
                             validation jobs through the shared job columns
  * training_runs.custom_model_id
                             which custom model a run trained on; RESTRICT, so a model in use can only be
                             archived, never removed from under its runs
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "plugin_settings",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("plugin_id", sa.Text(), nullable=False),
        sa.Column("task", sa.Text(), server_default="", nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("updated_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("kind", "plugin_id", "task", name="uq_plugin_settings_target"),
    )

    op.create_table(
        "custom_models",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), primary_key=True),
        sa.Column("owner_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("backend", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("source_kind", sa.String(16), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("revision", sa.String(64), nullable=True),
        sa.Column("storage_key", sa.Text(), nullable=True),
        sa.Column("sha256", sa.CHAR(64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("spec", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("tasks", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("status", sa.String(16), server_default="pending_upload", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        # The shared job columns (models.training.JobColumns): a custom model is claimed as a validation job.
        sa.Column("attempt", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="1", nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claimed_by", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    op.create_index("ix_custom_models_owner_user_id", "custom_models", ["owner_user_id"])
    op.create_index("ix_custom_models_status", "custom_models", ["status"])

    op.add_column(
        "training_runs",
        sa.Column("custom_model_id", sa.Uuid(), sa.ForeignKey("custom_models.id", ondelete="RESTRICT"), nullable=True),
    )
    op.create_index("ix_training_runs_custom_model_id", "training_runs", ["custom_model_id"])


def downgrade() -> None:
    op.drop_index("ix_training_runs_custom_model_id", table_name="training_runs")
    op.drop_column("training_runs", "custom_model_id")
    op.drop_table("custom_models")
    op.drop_table("plugin_settings")
    op.drop_column("users", "disabled_at")
