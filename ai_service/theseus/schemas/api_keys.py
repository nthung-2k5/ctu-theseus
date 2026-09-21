import uuid
from datetime import datetime

from pydantic import Field

from theseus.schemas.common import ApiModel


class CreateApiKeyBody(ApiModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyOut(ApiModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    last_used_at: datetime | None
    created_at: datetime
    revoked_at: datetime | None


class CreatedApiKey(ApiModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    created_at: datetime
    # The raw key: returned exactly once, never stored, not derivable from its hash.
    key: str


class ApiKeyListResponse(ApiModel):
    keys: list[ApiKeyOut]
