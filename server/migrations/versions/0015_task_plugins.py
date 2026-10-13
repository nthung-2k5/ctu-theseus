"""task plugins: `projects.task` becomes free text instead of the `project_task` enum

Revision ID: 0015
Revises: 0014

A task is a plugin class (`omnitrain/tasks/`), so the set of valid task ids is whatever is installed, not
something the database should enumerate; adding a task used to need an `ALTER TYPE ... ADD VALUE` migration
(see 0013). `exports.format` and `training_runs.backend` are free text for the same reason. The API validates
a new project's task against the registry; existing rows keep their ids unchanged.

The downgrade recreates the enum with the built-in tasks and casts back, so it fails (loudly, in the cast) if a
project already uses a task id that only a plugin defines.
"""

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

_BUILTIN_TASKS = (
    "text_classification", "token_classification", "text_generation", "question_answering",
    "summarization", "sequence_to_sequence", "text_embedding",
    "image_classification", "object_detection", "image_segmentation", "image_captioning",
    "audio_classification", "automatic_speech_recognition", "audio_segmentation", "audio_captioning",
    "tabular_regression", "tabular_classification", "tabular_clustering", "tabular_anomaly_detection",
    "video_classification",
)  # fmt: skip


def upgrade() -> None:
    op.alter_column("projects", "task", type_=sa.Text(), postgresql_using="task::text", existing_nullable=False)
    op.execute("DROP TYPE project_task")


def downgrade() -> None:
    values = ", ".join(f"'{t}'" for t in _BUILTIN_TASKS)
    op.execute(f"CREATE TYPE project_task AS ENUM ({values})")
    op.alter_column(
        "projects", "task", type_=sa.Enum(name="project_task", create_type=False),
        postgresql_using="task::project_task", existing_nullable=False,
    )  # fmt: skip
