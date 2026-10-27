"""The default trainer backend of a run or sweep is pytorch, not ludwig

Ludwig was removed. `training_runs.backend` and `sweeps.backend` stay free text (a plugin id); only the default a row
gets when nothing names a backend changes. The services always name one, so this is the schema agreeing with the code.
Rows already written keep their value.
"""

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("training_runs", "sweeps"):
        op.alter_column(table, "backend", server_default="pytorch")


def downgrade() -> None:
    for table in ("training_runs", "sweeps"):
        op.alter_column(table, "backend", server_default="ludwig")
