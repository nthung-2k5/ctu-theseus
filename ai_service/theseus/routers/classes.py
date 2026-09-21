"""/api/projects/{project_id}/classes: label classes (classification tasks).

Classes belong to the project 1:1 dataset. Deleting is soft (is_active=false) so annotations that
reference a class keep a valid foreign key.
"""

import uuid

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException

from theseus.db.models import LabelClass
from theseus.deps import ProjectDep, SessionDep
from theseus.schemas.projects import ClassListResponse, ClassResponse, CreateClassBody, LabelClassOut, UpdateClassBody

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


@router.get("", response_model=ClassListResponse)
async def list_classes(project: ProjectDep, session: SessionDep) -> ClassListResponse:
    rows = (
        (
            await session.execute(
                sa.select(LabelClass)
                .where(LabelClass.dataset_id == project.id, LabelClass.is_active.is_(True))
                .order_by(LabelClass.created_at, LabelClass.class_id)
            )
        )
        .scalars()
        .all()
    )
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
