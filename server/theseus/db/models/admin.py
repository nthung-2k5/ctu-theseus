import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db.base import Base
from theseus.db.models._common import updated_at, uuid_pk


class PluginSetting(Base):
    """An admin's enable/disable override for one plugin, model or task (see services/plugin_settings.py).

    A missing row means "enabled", so a fresh install behaves exactly as before. `kind` and
    `plugin_id` are free text (like exports.format), never a Postgres enum: a new plugin needs no migration.
    """

    __tablename__ = "plugin_settings"
    __table_args__ = (sa.UniqueConstraint("kind", "plugin_id", "task", name="uq_plugin_settings_target"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    # backend | builtin_model | export_format | preprocessing | augmentation | task
    kind: Mapped[str] = mapped_column(sa.Text)
    plugin_id: Mapped[str] = mapped_column(sa.Text)
    # '' means "every task"; a task id scopes the override to that task and beats the '' row.
    task: Mapped[str] = mapped_column(sa.Text, server_default="")
    enabled: Mapped[bool] = mapped_column(sa.Boolean)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(sa.ForeignKey("users.id", ondelete="SET NULL"))
    updated_at: Mapped[datetime] = updated_at()
