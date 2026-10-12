"""A bring-your-own model on a worker: fetch its files here, and point the run's config at them.

The server's compiled config names the model's files by an absolute path on the SERVER (`custom_models.local_dir_for`).
A worker keeps them at its own deterministic path instead, fetched with exactly the code the server uses
(`custom_models.ensure_local`: forbidden file types, zip-slip and size limits, the bundle checksum re-checked), and
swaps that prefix wherever it appears in the config.

The Hub is asked anonymously (`token=None` becomes `False` in `custom_models._hub_token`), so a token left in the
worker machine's environment is never used: the server hands a worker only a model that needs none.
"""

import uuid
from typing import Any

from omnitrain.db.models import CustomModel
from omnitrain.services import custom_models


def relocate(value: Any, old: str, new: str) -> Any:
    """`value` with every string starting with the path `old` rewritten to start with `new`, at any depth."""
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old) else value
    if isinstance(value, dict):
        return {k: relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    return value


def prepare(info: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Fetch the model described by a run spec's `customModel` and return `config` pointing at the local copy."""
    row = CustomModel(
        id=uuid.UUID(info["id"]),
        kind=info["kind"],
        source_kind=info["sourceKind"],
        source_ref=info.get("sourceRef"),
        revision=info.get("revision"),
        sha256=info.get("sha256"),
        spec=info.get("spec") or {},
    )
    local = custom_models.ensure_local(row, token=None)
    return relocate(config, info["serverPath"], str(local))
