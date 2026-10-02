"""What models a user can train on, for a backend and task: the built-ins an admin has not switched off,
plus the custom models they are allowed to see.

A custom model is visible to a user when ALL of these hold:
  * it belongs to this backend and lists this task;
  * it is `ready` (uploaded/pinned and validated), `enabled`, and not archived;
  * the backend still offers its kind for the task, and that kind is currently usable;
  * it is global (no owner: an admin's) or owned by that user.
Someone else's private model is indistinguishable from one that does not exist: `resolve` reports both
the same way, so an id cannot be used to probe for other people's models.

Custom models are addressed as "custom:{uuid}" in the hyperparameter the backend names for its model
choice (`encoderId` for Ludwig), which is why a built-in and a custom model share one picker.
"""

import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from theseus.backends.base import CustomModelKind, CustomModelRef, ModelChoice, TrainerBackend
from theseus.backends.registry import enabled_builtin_models
from theseus.db.models import CustomModel
from theseus.services import custom_models
from theseus.services.task_registry import TaskDescriptor

CUSTOM_PREFIX = "custom:"


class ModelNotAvailable(LookupError):
    """The requested custom model is not usable by this user for this task; the message is safe to show."""


def is_custom_id(model_id: str | None) -> bool:
    return bool(model_id) and model_id.startswith(CUSTOM_PREFIX)  # type: ignore[union-attr]


def _usable_kinds(backend: type[TrainerBackend], task: TaskDescriptor) -> dict[str, CustomModelKind]:
    return {k.id: k for k in backend.custom_model_kinds(task) if k.unavailable_reason is None}


def _visible(backend: type[TrainerBackend], task: TaskDescriptor, user_id: uuid.UUID) -> sa.ColumnElement[bool]:
    return sa.and_(
        CustomModel.backend == backend.id,
        CustomModel.tasks.any(task.id),
        CustomModel.status == "ready",
        CustomModel.enabled.is_(True),
        CustomModel.archived_at.is_(None),
        sa.or_(CustomModel.owner_user_id.is_(None), CustomModel.owner_user_id == user_id),
    )


async def _visible_rows(
    session: AsyncSession, user_id: uuid.UUID, task: TaskDescriptor, backend: type[TrainerBackend]
) -> list[CustomModel]:
    kinds = _usable_kinds(backend, task)
    if not kinds:
        return []
    stmt = sa.select(CustomModel).where(_visible(backend, task, user_id), CustomModel.kind.in_(kinds))
    return list((await session.execute(stmt.order_by(CustomModel.created_at, CustomModel.id))).scalars())


async def list_models(
    session: AsyncSession, user_id: uuid.UUID, task: TaskDescriptor, backend: type[TrainerBackend]
) -> list[ModelChoice]:
    """Built-ins first (as `backends.registry.describe` always listed them), then custom models."""
    models = enabled_builtin_models(backend, task)
    for row in await _visible_rows(session, user_id, task, backend):
        models.append(
            ModelChoice(
                id=f"{CUSTOM_PREFIX}{row.id}",
                label=row.name,
                description=row.description,
                pretrained=True,
                source="private" if row.owner_user_id is not None else "global",
                kind=row.kind,
            )
        )
    return models


async def resolve(
    session: AsyncSession,
    user_id: uuid.UUID,
    task: TaskDescriptor,
    backend: type[TrainerBackend],
    model_id: str,
) -> tuple[CustomModel, CustomModelRef]:
    """The custom model behind a "custom:{uuid}" id, if `user_id` may train `task` on it with `backend`."""
    try:
        wanted = uuid.UUID(model_id.removeprefix(CUSTOM_PREFIX))
    except ValueError:
        raise ModelNotAvailable(f"'{model_id}' is not a valid custom model id") from None
    for row in await _visible_rows(session, user_id, task, backend):
        if row.id == wanted:
            return row, custom_models.ref_for(row)
    raise ModelNotAvailable("That custom model does not exist or is not available for this task")
