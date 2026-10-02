"""drop label class display order

Revision ID: 0006
Revises: 0005

Label classes are listed in creation order again; the `position` column added in 0005 is unused.
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("label_classes", "position")


def downgrade() -> None:
    op.add_column("label_classes", sa.Column("position", sa.Integer(), nullable=False, server_default="0"))
