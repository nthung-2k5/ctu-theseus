import pytest
from pydantic import ValidationError

from theseus.db.base import Base
from theseus.services import storage
from theseus.settings import Settings


def test_pool_key_shards_by_hash_prefix():
    assert storage.pool_key("proj", "abcdef", ".png") == "pool/proj/ab/abcdef.png"


def test_key_layout_matches_documented_layout():
    assert storage.snapshot_parquet_key("v1") == "snapshots/v1/dataset.parquet"
    assert storage.training_config_key("r1") == "r1/config.yaml"
    assert storage.evaluation_report_key("r1") == "r1/evaluation/report.json"
    assert storage.bundle_key("r1", "e1") == "r1/bundles/e1.zip"
    assert storage.inference_upload_key("i1", ".csv") == "inference/i1/input.csv"


def test_async_database_url_adds_asyncpg_driver():
    s = Settings(CTU_THESEUS_DB_URI="postgres://u:p@h:5432/d")
    assert s.async_database_url == "postgresql+asyncpg://u:p@h:5432/d"


def test_production_requires_explicit_config(monkeypatch):
    for name in ("CTU_THESEUS_DB_URI", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "JWT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, ENVIRONMENT="production")


def test_metadata_has_every_table_and_load_bearing_constraints():
    import theseus.db.models  # noqa: F401

    tables = Base.metadata.tables
    assert {"users", "refresh_tokens", "run_events", "training_runs", "exports", "inference_jobs"} <= set(tables)
    # The better-auth tables are gone with the move to JWT.
    assert not {"sessions", "accounts", "verifications"} & set(tables)
    assert "conversion_job_id" not in tables["exports"].c
    index_names = {i.name for i in tables["annotations"].indexes}
    assert "annotations_item_classification_key" in index_names
    assert "confidence_bounds" in {c.name for c in tables["annotations"].constraints}
