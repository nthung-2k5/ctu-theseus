from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    """Base for every request/response model: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)


class ErrorBody(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    error: ErrorBody
