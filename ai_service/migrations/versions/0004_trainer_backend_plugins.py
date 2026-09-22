"""trainer backends as plugin ids

Revision ID: 0004
Revises: 0003

Ludwig was the only training framework, wired directly into jobs/services. Trainer backends are
now a plugin system (theseus/backends/), so `training_runs`/`sweeps` need a `backend` column (free
text, a plugin id, like exports.format) and `training_runs.ludwig_config` is renamed to `config`:
it's the compiled config from whichever backend the run names, opaque outside that backend.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("backend", sa.String(length=64), nullable=False, server_default="ludwig"))
    op.add_column("sweeps", sa.Column("backend", sa.String(length=64), nullable=False, server_default="ludwig"))
    op.alter_column("training_runs", "ludwig_config", new_column_name="config")


def downgrade() -> None:
    op.alter_column("training_runs", "config", new_column_name="ludwig_config")
    op.drop_column("sweeps", "backend")
    op.drop_column("training_runs", "backend")
