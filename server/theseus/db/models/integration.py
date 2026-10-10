import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from theseus.db.base import Base
from theseus.db.models._common import created_at, updated_at


class UserIntegration(Base):
    """A user's credential for an outside service, one row per (user, provider). Today only `huggingface`.

    The token is stored encrypted (`services/secrets.py`) and is never returned by the API: the UI only gets
    `token_hint` (its last characters) and `account_name`, both captured when it was saved. Deleting the
    user deletes the row.
    """

    __tablename__ = "user_integrations"

    user_id: Mapped[uuid.UUID] = mapped_column(sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    provider: Mapped[str] = mapped_column(sa.String(32), primary_key=True)
    token_encrypted: Mapped[str] = mapped_column(sa.Text)
    # The last few characters of the token, for the user to recognise which one is saved. Display only.
    token_hint: Mapped[str] = mapped_column(sa.String(16))
    # Who the service says the token belongs to (a Hugging Face username), at the time it was saved.
    account_name: Mapped[str | None] = mapped_column(sa.String(120))
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = updated_at()
