"""Admin enable/disable switches for plugins, built-in models and tasks.

The plugin registries are immutable after startup, so "disabled" is not a registry concept: it is a
filter the registries and a few routes apply through `is_enabled`. The switches live in the
`plugin_settings` table and are mirrored into a process-local snapshot, so `is_enabled` is a plain
synchronous dict lookup that registry code can call without a database session. That is safe because
the service runs as exactly one process (see lifespan.assert_single_process); the snapshot is loaded at
startup and replaced after every admin write.

Semantics
  * No row means enabled: a fresh install offers exactly what it always did.
  * A row scoped to a task (`task="image_classification"`) beats the row for every task (`task=""`), so an
    admin can switch something off everywhere and back on for one task, or the reverse.
  * Disabling only stops NEW use. Runs, exports and projects that already exist keep working, since
    nothing that reads a finished artifact consults these switches.
"""

import logging
import uuid
from typing import Literal, get_args

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.db.base import get_sessionmaker
from theseus.db.models import PluginSetting

logger = logging.getLogger(__name__)

PluginKind = Literal["backend", "builtin_model", "export_format", "preprocessing", "augmentation", "task"]
PLUGIN_KINDS: tuple[str, ...] = get_args(PluginKind)

_ALL_TASKS = ""

# (kind, plugin_id, task) -> enabled. Replaced wholesale, never mutated in place, so a reader never
# sees a half-applied update.
_state: dict[tuple[str, str, str], bool] = {}


def builtin_model_key(backend_id: str, model_id: str) -> str:
    """The `plugin_id` of a built-in model: the model id alone is not unique across backends."""
    return f"{backend_id}:{model_id}"


def is_enabled(kind: str, plugin_id: str, task: str | None = None) -> bool:
    """Whether `plugin_id` may be used (for `task`, when given). Defaults to True."""
    if task:
        scoped = _state.get((kind, plugin_id, task))
        if scoped is not None:
            return scoped
    return _state.get((kind, plugin_id, _ALL_TASKS), True)


def snapshot() -> dict[tuple[str, str, str], bool]:
    """A copy of every override, for the admin listing."""
    return dict(_state)


def reset() -> None:
    """Forget every override (tests)."""
    global _state
    _state = {}


async def refresh(session: AsyncSession) -> None:
    """Replace the snapshot with what the database says now."""
    global _state
    rows = (await session.execute(sa.select(PluginSetting))).scalars().all()
    _state = {(r.kind, r.plugin_id, r.task): r.enabled for r in rows}


async def load() -> None:
    """Startup: read the overrides before any request is served."""
    async with get_sessionmaker()() as session:
        await refresh(session)
    logger.info("Loaded %d plugin setting(s).", len(_state))


async def set_enabled(
    session: AsyncSession,
    *,
    kind: str,
    plugin_id: str,
    task: str | None,
    enabled: bool | None,
    user_id: uuid.UUID,
) -> None:
    """Write (or, with `enabled=None`, clear) one override and refresh the snapshot.

    Clearing an override makes the target inherit again: from the row for every task if this was a
    task-scoped override, otherwise from the default (enabled).
    """
    scope = task or _ALL_TASKS
    if enabled is None:
        await session.execute(
            sa.delete(PluginSetting).where(
                PluginSetting.kind == kind, PluginSetting.plugin_id == plugin_id, PluginSetting.task == scope
            )
        )
    else:
        stmt = pg_insert(PluginSetting).values(
            kind=kind, plugin_id=plugin_id, task=scope, enabled=enabled, updated_by=user_id
        )
        await session.execute(
            stmt.on_conflict_do_update(
                constraint="uq_plugin_settings_target",
                set_={"enabled": stmt.excluded.enabled, "updated_by": user_id, "updated_at": sa.func.now()},
            )
        )
    await session.commit()
    await refresh(session)
