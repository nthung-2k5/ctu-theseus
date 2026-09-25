from typing import Any, Literal

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


class ParamSpec(ApiModel):
    """One tunable parameter of a plugin (an augmentation op, a trainer backend's hyperparameter,
    ...), flattened so the web can render a form without knowing the plugin. See `theseus.params`."""

    name: str
    label: str
    description: str | None = None
    type: Literal["int", "float", "bool", "choice"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    # Section heading the form files this parameter under (e.g. "Optimisation"); parameters without
    # one are shown together under a generic heading. Order of first appearance is the section order.
    group: str | None = None
