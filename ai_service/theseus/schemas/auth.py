import uuid
from datetime import datetime

from pydantic import Field

from theseus.schemas.common import ApiModel

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class RegisterBody(ApiModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginBody(ApiModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class UserOut(ApiModel):
    id: uuid.UUID
    name: str
    email: str
    role: str
    created_at: datetime


class UserResponse(ApiModel):
    user: UserOut
