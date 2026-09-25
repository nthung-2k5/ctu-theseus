"""/api/projects/{project_id}/classes: label classes (classification tasks).

Classes belong to the project 1:1 dataset. Deleting is soft (is_active=false) so annotations that
reference a class keep a valid foreign key. They are listed in creation order.
"""

import uuid

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from theseus.db.models import LabelClass
from theseus.deps import ProjectDep, SessionDep
from theseus.schemas.projects import (
    ClassListResponse,
    ClassResponse,
    CreateClassBody,
    LabelClassOut,
    SaveClassesBody,
    UpdateClassBody,
)

router = APIRouter(prefix="/projects/{project_id}/classes", tags=["classes"])

# Default palette for auto-assigning colors to new classes.
CLASS_COLORS = [
    "#e03131", "#2f9e44", "#1971c2", "#f08c00", "#9c36b5", "#0c8599", "#e8590c", "#6741d9",
    "#3bc9db", "#ff6b6b", "#51cf66", "#339af0", "#fcc419", "#cc5de8", "#20c997", "#ff922b",
]  # fmt: skip


def _by_name(project_id: uuid.UUID, name: str) -> sa.Select:
    return sa.select(LabelClass).where(LabelClass.dataset_id == project_id, LabelClass.name == name)


async def _class_in_project(session, project_id: uuid.UUID, class_id: uuid.UUID, *, active_only: bool) -> LabelClass:
    cls = await session.get(LabelClass, class_id)
    if cls is None or (active_only and not cls.is_active):
        raise HTTPException(404, "Class not found")
    if cls.dataset_id != project_id:
        raise HTTPException(403, "Unauthorized")
    return cls


async def _active_classes(session, project_id: uuid.UUID) -> list[LabelClass]:
    rows = await session.execute(
        sa.select(LabelClass)
        .where(LabelClass.dataset_id == project_id, LabelClass.is_active.is_(True))
        .order_by(LabelClass.created_at, LabelClass.class_id)
    )
    return list(rows.scalars().all())


@router.get("", response_model=ClassListResponse)
async def list_classes(project: ProjectDep, session: SessionDep) -> ClassListResponse:
    rows = await _active_classes(session, project.id)
    return ClassListResponse(classes=[LabelClassOut.model_validate(c) for c in rows])


@router.post("", response_model=ClassResponse)
async def create_class(body: CreateClassBody, project: ProjectDep, session: SessionDep) -> ClassResponse:
    # Includes soft-deleted rows: (dataset_id, name) is unique across them.
    existing = (await session.execute(_by_name(project.id, body.name))).scalar_one_or_none()
    if existing is not None and existing.is_active:
        raise HTTPException(400, "Class name already exists")

    color = body.ui_color_hex
    if existing is not None:
        # Re-creating a soft-deleted class brings it back instead of violating the unique constraint.
        existing.is_active = True
        existing.description = body.description
        if color:
            existing.ui_color_hex = color
        cls = existing
    else:
        if not color:
            total = (
                await session.execute(sa.select(sa.func.count()).where(LabelClass.dataset_id == project.id))
            ).scalar_one()
            color = CLASS_COLORS[total % len(CLASS_COLORS)]
        cls = LabelClass(dataset_id=project.id, name=body.name, description=body.description, ui_color_hex=color)
        session.add(cls)
    await session.commit()
    await session.refresh(cls)
    return ClassResponse(class_=LabelClassOut.model_validate(cls))


@router.put("", response_model=ClassListResponse)
async def save_classes(body: SaveClassesBody, project: ProjectDep, session: SessionDep) -> ClassListResponse:
    """Replace the whole class list in one transaction: rename, recolor, describe, add and delete.

    `classes` is the desired final set:
      * an entry with `classId` updates that class (name, description, and colour when given);
      * an entry without one creates a class, or revives a deleted class of the same name;
      * an active class missing from the list is deleted (soft, like DELETE).
    Names must be unique ignoring case. Renames may swap or reuse names freely, including the name of a
    class deleted earlier. Concurrent saves serialize on a row lock: the last one wins.
    """
    rows = list(
        (await session.execute(sa.select(LabelClass).where(LabelClass.dataset_id == project.id).with_for_update()))
        .scalars()
        .all()
    )
    by_id = {c.class_id: c for c in rows}

    seen_ids: set[uuid.UUID] = set()
    seen_names: set[str] = set()
    for entry in body.classes:
        if entry.class_id is not None:
            cls = by_id.get(entry.class_id)
            if cls is None or not cls.is_active:
                raise HTTPException(404, f"Class not found: {entry.class_id}")
            if entry.class_id in seen_ids:
                raise HTTPException(400, f"Class listed more than once: {entry.class_id}")
            seen_ids.add(entry.class_id)
        folded = entry.name.casefold()
        if folded in seen_names:
            raise HTTPException(400, f'Duplicate class name "{entry.name}"')
        seen_names.add(folded)

    # Rows that are not in the final list: earlier soft-deletes, plus the ones this save deletes.
    dropped = [c for c in rows if c.class_id not in seen_ids]
    for cls in dropped:
        cls.is_active = False
    dropped_by_name = {c.name: c for c in dropped}

    # Pair every entry with its row: by id, else by reviving a dropped row of the exact same name.
    revived: set[uuid.UUID] = set()
    plan: list[tuple] = []
    for entry in body.classes:
        cls = by_id[entry.class_id] if entry.class_id is not None else None
        if cls is None:
            tombstone = dropped_by_name.get(entry.name)
            if tombstone is not None and tombstone.class_id not in revived:
                cls = tombstone
                revived.add(tombstone.class_id)
        plan.append((entry, cls))

    # (dataset_id, name) is unique and checked per statement, so free every name that is about to move
    # first: renamed classes get a temporary name (lets two classes swap names), and a deleted class
    # sitting on a name someone else now wants is renamed out of the way.
    for entry, cls in plan:
        if cls is not None and cls.name != entry.name:
            cls.name = f"~{cls.class_id}"
    wanted = {entry.name for entry in body.classes}
    for tombstone in dropped:
        if tombstone.class_id not in revived and tombstone.name in wanted:
            tombstone.name = f"{tombstone.name[:80]} (deleted {str(tombstone.class_id)[:8]})"
    await session.flush()

    created = len(rows)
    for entry, cls in plan:
        if cls is None:
            color = entry.ui_color_hex or CLASS_COLORS[created % len(CLASS_COLORS)]
            created += 1
            session.add(
                LabelClass(
                    dataset_id=project.id,
                    name=entry.name,
                    description=entry.description,
                    ui_color_hex=color,
                )
            )
            continue
        cls.name = entry.name
        cls.description = entry.description
        if entry.ui_color_hex:
            cls.ui_color_hex = entry.ui_color_hex
        cls.is_active = True
    await session.commit()
    return ClassListResponse(
        classes=[LabelClassOut.model_validate(c) for c in await _active_classes(session, project.id)]
    )


@router.patch("/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: uuid.UUID, body: UpdateClassBody, project: ProjectDep, session: SessionDep
) -> ClassResponse:
    cls = await _class_in_project(session, project.id, class_id, active_only=True)
    if body.name is not None and body.name != cls.name:
        if (await session.execute(_by_name(project.id, body.name))).scalar_one_or_none() is not None:
            raise HTTPException(400, "Class name already exists")
    for field in body.model_fields_set:
        value = getattr(body, field)
        if field == "name" and value is None:
            continue
        setattr(cls, field, value)
    await session.commit()
    await session.refresh(cls)
    return ClassResponse(class_=LabelClassOut.model_validate(cls))


@router.delete("/{class_id}", status_code=204)
async def delete_class(class_id: uuid.UUID, project: ProjectDep, session: SessionDep) -> None:
    cls = await _class_in_project(session, project.id, class_id, active_only=False)
    cls.is_active = False
    await session.commit()
