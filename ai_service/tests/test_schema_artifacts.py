"""The committed schema/*.json files must match what the code generates (CI drift check)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_schema.py"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_schema", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_artifacts_are_up_to_date():
    for name, expected in _load_script().artifacts().items():
        actual = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        assert actual == expected, (
            f"schema/{name} is stale. Run: cd ai_service && uv run python scripts/export_schema.py"
        )


def test_importing_the_app_needs_no_infrastructure(monkeypatch):
    # create_app() runs at export time with no Postgres or S3 reachable.
    monkeypatch.setenv("CTU_THESEUS_DB_URI", "postgres://nobody:nothing@203.0.113.1:1/none")
    monkeypatch.setenv("S3_ENDPOINT", "http://203.0.113.1:1")
    from theseus.app import create_app

    assert create_app().openapi()["info"]["title"] == "CTU Theseus API"
