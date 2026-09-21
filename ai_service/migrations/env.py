"""Async Alembic environment. The DB URL comes from theseus.settings, never from alembic.ini."""

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

import theseus.db.models  # noqa: F401  (registers every table on Base.metadata)
from theseus.db.base import Base
from theseus.settings import get_settings

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(get_settings().async_database_url)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
