import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db.base import Base
from theseus.db.models._common import created_at, updated_at, uuid_pk


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(sa.Text)
    email: Mapped[str] = mapped_column(sa.Text, unique=True)
    password_hash: Mapped[str] = mapped_column(sa.Text)
    role: Mapped[str] = mapped_column(sa.Text, server_default="user")
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()


class RefreshToken(Base):
    """Opaque, hashed, rotated on use. family_id groups a rotation chain so a replayed token revokes all of it."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    family_id: Mapped[uuid.UUID] = mapped_column(sa.Uuid, index=True)
    token_hash: Mapped[str] = mapped_column(sa.CHAR(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at()
    revoked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    replaced_by: Mapped[uuid.UUID | None] = mapped_column(sa.Uuid)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(sa.String(100))
    # sha256 hex digest of the raw key, never the raw key itself.
    key_hash: Mapped[str] = mapped_column(sa.CHAR(64), unique=True)
    # First few characters of the raw key, display only.
    key_prefix: Mapped[str] = mapped_column(sa.String(16))
    last_used_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at()
    # Soft-revoked, keeping the audit trail of what a (possibly leaked) key was.
    revoked_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
