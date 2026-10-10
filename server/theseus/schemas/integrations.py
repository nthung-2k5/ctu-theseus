"""Request/response models for the outside-service integrations in Settings."""

from datetime import datetime

from pydantic import Field, field_validator

from theseus.schemas.common import ApiModel


class HuggingFaceStatus(ApiModel):
    """Whether the caller has saved a Hugging Face token. The token itself is never sent back."""

    connected: bool
    # The Hugging Face username the token belongs to, as reported when it was saved.
    account_name: str | None = None
    # The last few characters of the saved token, so the user can tell which one it is.
    token_hint: str | None = None
    connected_at: datetime | None = None


class ConnectHuggingFaceBody(ApiModel):
    token: str = Field(min_length=8, max_length=512)

    @field_validator("token")
    @classmethod
    def _looks_like_a_token(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("hf_") or any(c.isspace() for c in v):
            raise ValueError(
                "A Hugging Face access token starts with hf_ (create one at huggingface.co/settings/tokens)"
            )
        return v
