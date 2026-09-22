"""snapshot-time augmentation columns

Revision ID: 0003
Revises: 0002

Augmented items are real dataset_items rows pointing at their source item. The (dataset_id,
content_hash) uniqueness becomes partial (originals only), so an upload can never dedup onto an
augmented copy.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- dataset items: augmented copies ------------------------------------------------------
    op.add_column("dataset_items", sa.Column("source_item_id", sa.Uuid(), nullable=True))
    op.add_column("dataset_items", sa.Column("augmentation", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.create_foreign_key(
        "dataset_items_sourceItemId_fkey", "dataset_items", "dataset_items", ["source_item_id"], ["id"], ondelete="RESTRICT"
    )
    op.create_index("ix_dataset_items_source_item_id", "dataset_items", ["source_item_id"], unique=False)
    op.drop_constraint("dataset_items_datasetId_contentHash_key", "dataset_items", type_="unique")
    op.create_index(
        "dataset_items_pool_contentHash_key",
        "dataset_items",
        ["dataset_id", "content_hash"],
        unique=True,
        postgresql_where=sa.text("source_item_id IS NULL"),
    )

    # -- dataset versions: what the snapshot was built with -----------------------------------
    op.add_column(
        "dataset_versions", sa.Column("augmentation_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )
    op.add_column("dataset_versions", sa.Column("augmented_count", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("dataset_versions", "augmented_count")
    op.drop_column("dataset_versions", "augmentation_config")

    # Augmented items cannot exist under the old single unique constraint's meaning: drop them first.
    op.execute("DELETE FROM dataset_version_items WHERE item_id IN (SELECT id FROM dataset_items WHERE source_item_id IS NOT NULL)")
    op.execute("DELETE FROM annotations WHERE item_id IN (SELECT id FROM dataset_items WHERE source_item_id IS NOT NULL)")
    for table in ("text_features", "vision_features", "audio_features", "tabular_features"):
        op.execute(f"DELETE FROM {table} WHERE item_id IN (SELECT id FROM dataset_items WHERE source_item_id IS NOT NULL)")
    op.execute("DELETE FROM dataset_items WHERE source_item_id IS NOT NULL")
    op.drop_index("dataset_items_pool_contentHash_key", table_name="dataset_items")
    op.create_unique_constraint(
        "dataset_items_datasetId_contentHash_key", "dataset_items", ["dataset_id", "content_hash"]
    )
    op.drop_index("ix_dataset_items_source_item_id", table_name="dataset_items")
    op.drop_constraint("dataset_items_sourceItemId_fkey", "dataset_items", type_="foreignkey")
    op.drop_column("dataset_items", "augmentation")
    op.drop_column("dataset_items", "source_item_id")
