"""label class display order

Revision ID: 0005
Revises: 0004

Added a `position` column so the class list could be reordered. The feature was dropped again in
0006, but 0005 stays in the chain: databases that already ran it record revision 0005, and Alembic
cannot upgrade a database whose current revision is missing from the scripts.
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("label_classes", sa.Column("position", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("label_classes", "position")
