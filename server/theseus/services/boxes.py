"""Bounding-box annotations: what one looks like, how it is validated, and how a snapshot flattens them.

An item's boxes are one `annotations` row per box, `annotation_type = 'bounding_box'`, with `class_id` set.
`label_structured` is `{"x", "y", "w", "h"}`: the top-left corner and the size, each a fraction of the whole
image (0..1) measured on the image as it is *displayed* (EXIF orientation applied). Fractions rather than
pixels, so a box survives a thumbnail, a resize or a re-upload, and nothing at training time needs the
image's dimensions.

An image with no objects is a first-class training example (a detector learns what NOT to fire on), and it
must not be mistaken for an image nobody labelled yet. It is a single marker row: `bounding_box`, no class,
`label_structured = {"background": true}`.
"""

import json
import uuid
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Boxes are stored as floats and compared after a round trip through JSON; allow for the last bit.
_EDGE_TOLERANCE = 1e-6


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_the_image(self) -> "BoundingBox":
        if self.x + self.w > 1 + _EDGE_TOLERANCE or self.y + self.h > 1 + _EDGE_TOLERANCE:
            raise ValueError("The box extends past the edge of the image")
        return self


class BackgroundMarker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    background: Literal[True]


def is_background(label_structured: Any) -> bool:
    return isinstance(label_structured, dict) and label_structured.get("background") is True


def validate_box_annotation(class_id: uuid.UUID | None, label_structured: Any) -> None:
    """Raise ValueError (with a user-facing message) unless this is a well-formed box, or a well-formed marker."""
    try:
        if is_background(label_structured):
            BackgroundMarker.model_validate(label_structured)
            if class_id is not None:
                raise ValueError("A 'no objects' marker has no class")
            return
        if class_id is None:
            raise ValueError("A bounding box needs a class")
        BoundingBox.model_validate(label_structured)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ValueError(f"Invalid bounding box{f' ({where})' if where else ''}: {first['msg']}") from None


def boxes_json(
    annotations: Iterable[tuple[uuid.UUID | None, Any]], class_name_by_id: dict[uuid.UUID, str]
) -> str | None:
    """The snapshot's `boxes` cell for one item, from its `bounding_box` annotations as (class_id, label_structured).

    A JSON list of `{label, x, y, w, h}`, labelled by class NAME (like the classification `class` column). `"[]"`
    means "labelled, and there is nothing in the image"; None means "not labelled", so a trainer leaves the item
    out instead of teaching the model that it is empty. A box of a class that has since been deleted is dropped,
    and an item left with nothing but such boxes is therefore unlabelled again.
    """
    boxes: list[dict[str, Any]] = []
    marked_empty = False
    for class_id, label in annotations:
        if is_background(label):
            marked_empty = True
        elif class_id in class_name_by_id and isinstance(label, dict):
            boxes.append({"label": class_name_by_id[class_id], **{k: label[k] for k in ("x", "y", "w", "h")}})
    if boxes:
        return json.dumps(boxes, separators=(",", ":"))
    return "[]" if marked_empty else None


def parse_boxes(cell: Any) -> list[dict[str, Any]] | None:
    """The inverse of `boxes_json` for a trainer: None for an unlabelled item, else the list (possibly empty)."""
    if not isinstance(cell, str):
        return None
    return json.loads(cell)
