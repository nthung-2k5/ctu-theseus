"""The model source plugin base class: where a bring-your-own model's files come from.

Two ship today: a Hugging Face Hub repository fetched at a pinned commit (`hub.py`) and a zip the user uploads
(`upload.py`). To add one (ModelScope, a URL, another bucket), create a module in this package with a subclass
that sets `id` and `label` and implements `fetch` and `validate`. Restart the backend; the id is then accepted
wherever a custom model names its `source_kind` (a trainer backend still lists the sources a kind may use, in
`CustomModelKind.source_kinds`). The web's import form only knows how to collect a Hub id or a file, so a new
source needs a form there to be offered to people; the backend side needs nothing else.

What a source never gets to decide, because it is the same for every source and lives in
`services/custom_models.py`: the deterministic local path, the completion marker, `check_tree`'s refusal of
pickles, code and symlinks, and the size limit. `fetch` puts files at a path it is given; `validate` runs the
first contact (pinning, first download, hashing) and hands the result back for `check_tree` to judge.

A worker (`omnitrain/worker/`) runs the same `fetch`, with no database and no secret. A source is only offered
to workers when `worker_fetchable` says so, and a worker must run the same package version, so keep it False
unless `fetch` needs nothing but S3 and the public internet.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa

from omnitrain.core.plugins import Registry

if TYPE_CHECKING:
    from pathlib import Path

    from omnitrain.db.models import CustomModel

_registry: Registry["ModelSource"] = Registry("model source", "omnitrain.plugins.model_sources")


class SourceRefError(ValueError):
    """The reference a user gave for a model is not one this source accepts; the message is safe to show."""


@dataclass
class Validated:
    """What the validation job learns on first contact with a model, to persist on its row."""

    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    # Needs its owner's credential to read (a private or gated Hub model). Lives in the row's `spec`.
    private: bool | None = None


class ModelSource:
    id: ClassVar[str]
    label: ClassVar[str]
    # The user must name the model (a repository id) vs. the files arrive another way (an upload).
    needs_ref: ClassVar[bool] = False
    # The model is created waiting for the browser to upload a bundle, then goes through upload-url/finalize.
    needs_upload: ClassVar[bool] = False
    # Added to a validation failure of a model shared with everyone, which never carries one person's credential.
    shared_failure_hint: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "id" in cls.__dict__ and not hasattr(cls, "label"):
            raise TypeError(f"Model source {cls.id!r} ({cls.__module__}) needs a label")
        _registry.register(cls)

    @classmethod
    def normalize_ref(cls, ref: str | None) -> str | None:
        """The cleaned `source_ref` to store, or `SourceRefError`. The default takes no reference at all."""
        if (ref or "").strip():
            raise SourceRefError("An uploaded model has no source reference")
        return None

    @classmethod
    async def token_for(cls, owner_user_id: uuid.UUID | None) -> str | None:
        """The owner's saved credential for this source, if it uses one. None for a model shared with everyone
        (it has no owner) and for a source that needs none; never anyone else's and never a server-wide one."""
        return None

    @classmethod
    def fetch(cls, row: "CustomModel", dest: "Path", token: str | None) -> None:
        """Put the model's files in `dest` (an empty directory). Raises `CustomModelFileError` when it cannot."""
        raise NotImplementedError

    @classmethod
    def validate(cls, row: "CustomModel", token: str | None, limit: int) -> tuple[Validated, "Path"]:
        """First contact: pin or hash the model, bring its files to disk, and return what to persist and the
        directory holding them. Mutates the detached `row` only to carry the pin forward (`revision`, `sha256`)."""
        raise NotImplementedError

    @classmethod
    def worker_fetchable(cls, model: Any) -> Any:
        """SQL condition (over the aliased `CustomModel` entity `model`): a worker may fetch this model by itself,
        with no secret. The queue uses it to decide which runs a remote worker may claim."""
        return sa.false()


def get_source(source_id: str) -> type[ModelSource]:
    """The source for an id. Raises KeyError('Unknown model source ...') if none is installed."""
    return _registry.get(source_id)


def find_source(source_id: str) -> type[ModelSource] | None:
    return _registry.find(source_id)


def list_sources() -> list[type[ModelSource]]:
    return sorted(_registry.all(), key=lambda s: s.id)
