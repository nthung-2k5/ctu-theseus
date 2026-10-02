import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db import enums as e
from theseus.db.base import Base
from theseus.db.models._common import created_at, pg_enum, updated_at, uuid_pk
from theseus.db.models.training import JobColumns

export_status_t = pg_enum(e.ExportStatus, "export_status")


class ModelExport(JobColumns, Base):
    """One export bundle request.

    Status flow: pending (queued) -> converting -> assembling -> ready | failed. A single
    export job walks the whole flow: convert the model if its artifact is missing, then zip.
    """

    __tablename__ = "exports"

    # Retryable: a failed attempt is re-queued after a delay, up to this many attempts.
    max_attempts: Mapped[int] = mapped_column(sa.Integer, server_default="3")

    id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"))
    # An export format plugin id (theseus/export/formats/). Deliberately free text, not a Postgres
    # enum: adding a format is a new Python class, never a migration.
    format: Mapped[str] = mapped_column(sa.String(64))
    status: Mapped[str] = mapped_column(export_status_t, server_default="pending", index=True)
    bundle_key: Mapped[str | None] = mapped_column(sa.Text)
    byte_size: Mapped[int | None] = mapped_column(sa.Integer)
    checksum: Mapped[str | None] = mapped_column(sa.CHAR(64))
    failed_message: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = created_at()
    ready_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    updated_at: Mapped[datetime] = updated_at()
