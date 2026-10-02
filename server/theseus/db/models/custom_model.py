import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db.base import Base
from theseus.db.models._common import created_at, updated_at, uuid_pk
from theseus.db.models.training import JobColumns


class CustomModel(JobColumns, Base):
    """A bring-your-own model: a Hugging Face / timm Hub reference or an uploaded weights bundle.

    `owner_user_id` NULL means a global model an admin manages for everyone; otherwise it is private to
    that user. A run that trains on one records it in `training_runs.custom_model_id` (ON DELETE
    RESTRICT), so deleting a model that any run used archives it instead: its files stay, and the runs
    that trained on it keep loading and exporting.

    It is claimed as a validation job (see jobs/validate_model.py): the shared job columns come from
    `JobColumns`, and `last_error` there is what the UI shows for a failed model.

    Everything that identifies WHAT the model is (`backend`, `kind`, `source_*`, `revision`, `sha256`) is
    fixed once created. Only the name, description, task list and switches change, so a run's record of
    "which model" never silently changes meaning under it.
    """

    __tablename__ = "custom_models"

    id: Mapped[uuid.UUID] = uuid_pk()
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("users.id", ondelete="SET NULL"))
    # A trainer backend plugin id, and one of that backend's `custom_model_kinds`. Free text, like
    # training_runs.backend: a new backend or kind needs no migration.
    backend: Mapped[str] = mapped_column(sa.String(64))
    kind: Mapped[str] = mapped_column(sa.String(64))
    name: Mapped[str] = mapped_column(sa.String(120))
    description: Mapped[str] = mapped_column(sa.Text, server_default="")
    # hub | upload
    source_kind: Mapped[str] = mapped_column(sa.String(16))
    # Hub repo / model name (hub only).
    source_ref: Mapped[str | None] = mapped_column(sa.Text)
    # The pinned Hub commit sha, resolved at validation. What makes a Hub model reproducible.
    revision: Mapped[str | None] = mapped_column(sa.String(64))
    # theseus-models key of the uploaded bundle (upload only).
    storage_key: Mapped[str | None] = mapped_column(sa.Text)
    sha256: Mapped[str | None] = mapped_column(sa.CHAR(64))
    size_bytes: Mapped[int | None] = mapped_column(sa.BigInteger)
    # Backend-specific extras, free-form.
    spec: Mapped[Any] = mapped_column(JSONB, server_default=sa.text("'{}'::jsonb"))
    # The tasks this model is offered for.
    tasks: Mapped[list[str]] = mapped_column(ARRAY(sa.Text), server_default=sa.text("'{}'::text[]"))
    # pending_upload | uploaded | validating | ready | failed
    status: Mapped[str] = mapped_column(sa.String(16), server_default="pending_upload", index=True)
    enabled: Mapped[bool] = mapped_column(sa.Boolean, server_default=sa.true())
    archived_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
