"""snapshot-time preprocessing

Revision ID: 0008
Revises: 0007

Preprocessing sits alongside augmentation as a second snapshot-time transform: deterministic ops,
each scoped to the splits (train/validation/test) it runs on. A preprocessed item is a real
dataset_items row that REPLACES its original in the snapshot's membership for those splits, rather
than adding a copy. `dataset_items.preprocessing` records what produced it, and is carried forward
onto an augmented copy made from a preprocessed train item.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dataset_items", sa.Column("preprocessing", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column(
        "dataset_versions", sa.Column("preprocessing_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )
    op.add_column(
        "dataset_versions", sa.Column("preprocessed_count", sa.Integer(), server_default="0", nullable=False)
    )


def downgrade() -> None:
    op.drop_column("dataset_versions", "preprocessed_count")
    op.drop_column("dataset_versions", "preprocessing_config")
    op.drop_column("dataset_items", "preprocessing")
