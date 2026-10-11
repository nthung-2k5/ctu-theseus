import uuid
from datetime import datetime
from typing import Any, get_args

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column


def pg_enum(literal: Any, name: str) -> ENUM:
    """A native Postgres enum built from a Literal alias (one CREATE TYPE, shared across tables)."""
    return ENUM(*get_args(literal), name=name, create_type=True)


def uuid_pk() -> Mapped[uuid.UUID]:
    # uuidv7() is a Postgres-side default (needs PG18) so time-ordered ids are assigned by the DB.
    return mapped_column(sa.Uuid, primary_key=True, server_default=sa.text("uuidv7()"))


def created_at() -> Mapped[datetime]:
    return mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())


def updated_at() -> Mapped[datetime]:
    return mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
