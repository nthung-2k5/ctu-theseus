"""A Hugging Face Hub repository, fetched at a pinned commit.

It may be private or gated: it is then fetched with its OWNER's saved Hugging Face token and nobody else's
(`services/integrations.py`). The Hub client itself lives in `services/custom_models.py`.
"""

import re
import uuid

import sqlalchemy as sa

from omnitrain.model_sources.base import ModelSource, SourceRefError, Validated
from omnitrain.services import custom_models

_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*(/[\w.\-]+)?$")


class Hub(ModelSource):
    id = "hub"
    label = "Hugging Face Hub"
    needs_ref = True
    shared_failure_hint = "Models shared with everyone can only come from public repositories."

    @classmethod
    def normalize_ref(cls, ref):
        ref = (ref or "").strip()
        if not _REPO_ID.match(ref):
            raise SourceRefError("Enter the model's Hugging Face repository id (e.g. bert-base-uncased or org/name)")
        return ref

    @classmethod
    async def token_for(cls, owner_user_id: uuid.UUID | None) -> str | None:
        # Imported here so a worker, which never has anyone's token, does not load the credential stack.
        from omnitrain.services import integrations

        return await integrations.huggingface_token_for(owner_user_id)

    @classmethod
    def fetch(cls, row, dest, token):
        custom_models._fetch_hub(row, dest, token)

    @classmethod
    def validate(cls, row, token, limit):
        pin = custom_models.hub_pin(row.source_ref, token)
        if pin.size_bytes > limit:
            raise custom_models.CustomModelFileError(
                f"The model is {pin.size_bytes // 2**20} MiB, over the {limit // 2**20} MiB limit"
            )
        row.revision = pin.revision
        path = custom_models.ensure_local(row, token)
        return Validated(revision=pin.revision, private=pin.private), path

    @classmethod
    def worker_fetchable(cls, model):
        # A private or gated model needs its owner's token, which a worker never gets.
        return sa.func.coalesce(model.spec["private"].as_boolean(), False).is_(False)
