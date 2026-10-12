"""video modality: `video` modality, `video_classification` task, `video_features` table

Revision ID: 0013
Revises: 0012

`modality` and `project_task` are native Postgres enums, so a new value is an `ALTER TYPE ... ADD VALUE`. That
statement cannot run inside a transaction block on older servers and a new value cannot be used until the
statement has committed, so each runs in an autocommit block of its own. Enum values cannot be dropped, so the
downgrade removes the table and the type it owns but leaves the two added values in place (harmless: nothing
references them once the video rows are gone).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE modality ADD VALUE IF NOT EXISTS 'video'")
        op.execute("ALTER TYPE project_task ADD VALUE IF NOT EXISTS 'video_classification'")

    video_container = postgresql.ENUM("mp4", "webm", name="video_container", create_type=False)
    video_container.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "video_features",
        sa.Column("item_id", sa.Uuid(), sa.ForeignKey("dataset_items.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Numeric(10, 3), nullable=False),
        sa.Column("fps", sa.Numeric(7, 3), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("container", video_container, nullable=True),
        sa.Column("codec", sa.String(16), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("video_features")
    postgresql.ENUM(name="video_container").drop(op.get_bind(), checkfirst=True)
