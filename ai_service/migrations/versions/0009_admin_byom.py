"""admin area: account disabling

Revision ID: 0009
Revises: 0008

Adds `users.disabled_at`: an admin can disable an account, which then cannot sign in or refresh.
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "disabled_at")
