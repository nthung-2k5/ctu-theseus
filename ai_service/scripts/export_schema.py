"""Generate the committed cross-language artifacts in the repo-level schema/ directory.

  schema/task_registry.json   the task registry, for the web task picker and per-modality UI
  schema/openapi.json         the API contract, the input to Orval

Run on the host, never inside the container (./schema is bind-mounted read-only there):

    cd ai_service && uv run python scripts/export_schema.py            # write
    cd ai_service && uv run python scripts/export_schema.py --check    # CI: fail if stale

Importing the app must not touch Postgres or S3; that is a design constraint of create_app().
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theseus.app import create_app  # noqa: E402
from theseus.services.task_registry import registry_json  # noqa: E402

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def render(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def artifacts() -> dict[str, str]:
    return {
        "task_registry.json": render(registry_json()),
        "openapi.json": render(create_app().openapi()),
    }


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = []
    for name, content in artifacts().items():
        path = SCHEMA_DIR / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if check:
            stale.append(name)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    if stale:
        print(
            f"Stale generated files: {', '.join(stale)}. Run: uv run python scripts/export_schema.py", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
