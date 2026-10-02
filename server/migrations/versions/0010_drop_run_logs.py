"""drop training run logs

Revision ID: 0010
Revises: 0009

Training runs no longer capture or stream their log: the run page has no Logs tab and nothing writes
`log` events. Delete the stored log rows and remove `log` from the `run_event_kind` enum (Postgres cannot
drop one enum value, so the type is recreated). The downgrade restores the value; the deleted rows are gone
for good.
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def _retype_run_event_kind(values: str) -> None:
    op.execute("ALTER TYPE run_event_kind RENAME TO run_event_kind_old")
    op.execute(f"CREATE TYPE run_event_kind AS ENUM ({values})")
    op.execute("ALTER TABLE run_events ALTER COLUMN kind TYPE run_event_kind USING kind::text::run_event_kind")
    op.execute("DROP TYPE run_event_kind_old")


def upgrade() -> None:
    op.execute("DELETE FROM run_events WHERE kind = 'log'")
    _retype_run_event_kind("'status', 'metric'")


def downgrade() -> None:
    _retype_run_event_kind("'status', 'metric', 'log'")
