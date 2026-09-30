"""admin area: account disabling and availability switches

Revision ID: 0009
Revises: 0008

  * users.disabled_at        an admin can disable an account
  * plugin_settings          admin enable/disable switches for plugins, built-in models and tasks; no row
                             means enabled, so an existing install behaves exactly as before
"""

import sqlalchemy as sa
from alembic import op

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



def downgrade() -> None:
    op.drop_table("plugin_settings")
    op.drop_column("users", "disabled_at")
