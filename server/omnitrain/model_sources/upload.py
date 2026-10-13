"""An uploaded zip, kept in `theseus-models/custom/{modelId}/`. Anyone can fetch it: it is read from S3."""

import sqlalchemy as sa

from omnitrain.model_sources.base import ModelSource, Validated
from omnitrain.services import custom_models


class Upload(ModelSource):
    id = "upload"
    label = "Uploaded bundle"
    needs_upload = True

    @classmethod
    def fetch(cls, row, dest, token):
        # Looked up on the module at call time: tests (and the worker's) replace `custom_models._fetch_upload`.
        custom_models._fetch_upload(row, dest)

    @classmethod
    def validate(cls, row, token, limit):
        imported = custom_models.import_upload(row)
        row.sha256 = imported.sha256
        return Validated(sha256=imported.sha256), imported.path

    @classmethod
    def worker_fetchable(cls, model):
        return sa.true()
