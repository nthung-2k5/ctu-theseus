"""sweep strategy plugins: `sweeps.strategy` becomes free text instead of the `sweep_strategy` enum

Revision ID: 0016
Revises: 0015

A sweep strategy is a plugin class (`omnitrain/sweep_strategies/`), so the set of valid ids is whatever is
installed, not something the database should enumerate (same reasoning as 0015 for `projects.task`). The API
validates a new sweep's strategy against the registry; existing rows keep their ids unchanged.

The downgrade recreates the enum with `grid` and `random` and casts back, so it fails (loudly, in the cast) if a
sweep already uses a strategy that only a plugin defines.
"""

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "sweeps", "strategy", type_=sa.String(64), postgresql_using="strategy::text", existing_nullable=False
    )
    op.execute("DROP TYPE sweep_strategy")


def downgrade() -> None:
    op.execute("CREATE TYPE sweep_strategy AS ENUM ('grid', 'random')")
    op.alter_column(
        "sweeps", "strategy", type_=sa.Enum(name="sweep_strategy", create_type=False),
        postgresql_using="strategy::sweep_strategy", existing_nullable=False,
    )  # fmt: skip
