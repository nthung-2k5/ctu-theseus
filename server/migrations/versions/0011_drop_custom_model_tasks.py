"""a custom model is offered for every task its kind serves

Revision ID: 0011
Revises: 0010

`custom_models.tasks` goes: a model's kind already decides which tasks it can be used for, and the backend
says which those are, so a per-model list could only drift from it (or hide a model from tasks it works
for). Nothing reads the column any more. The downgrade restores it, filled from the kind as the kinds stood
when this migration was written.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("custom_models", "tasks")


def downgrade() -> None:
    op.add_column(
        "custom_models",
        sa.Column("tasks", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
    )
    op.execute(
        """
        UPDATE custom_models SET tasks = CASE kind
            WHEN 'hf_transformer' THEN ARRAY['text_classification']
            WHEN 'hf_causal_lm' THEN ARRAY['text_generation', 'summarization', 'sequence_to_sequence', 'question_answering']
            WHEN 'hf_vision' THEN ARRAY['image_classification']
            WHEN 'timm_image' THEN ARRAY['image_classification', 'image_captioning']
            ELSE ARRAY[]::text[]
        END
        """
    )
