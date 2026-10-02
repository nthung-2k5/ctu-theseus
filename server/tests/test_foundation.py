from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from theseus.db.base import Base
from theseus.services import storage
from theseus.settings import Settings, get_settings


def test_pool_key_shards_by_hash_prefix():
    assert storage.pool_key("proj", "abcdef", ".png") == "pool/proj/ab/abcdef.png"


def test_key_layout_matches_documented_layout():
    assert storage.snapshot_parquet_key("v1") == "snapshots/v1/dataset.parquet"
    assert storage.training_config_key("r1") == "r1/config.yaml"
    assert storage.evaluation_report_key("r1") == "r1/evaluation/report.json"
    assert storage.bundle_key("r1", "e1") == "r1/bundles/e1.zip"


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
    assert {"users", "refresh_tokens", "run_events", "training_runs", "exports"} <= set(tables)
    assert "inference_jobs" not in tables  # predictions run in the request; nothing is stored
    # The better-auth tables are gone with the move to JWT.
    assert not {"sessions", "accounts", "verifications"} & set(tables)
    assert "conversion_job_id" not in tables["exports"].c
    index_names = {i.name for i in tables["annotations"].indexes}
    assert "annotations_item_classification_key" in index_names
    assert "confidence_bounds" in {c.name for c in tables["annotations"].constraints}


@pytest.fixture
def fresh_s3_clients():
    """`storage.s3` caches clients per endpoint, so a test that changes the settings must not see (or leave
    behind) a client built for another test's endpoint."""
    storage.s3.cache_clear()
    yield
    storage.s3.cache_clear()


def test_download_urls_are_signed_for_the_browser_facing_endpoint_not_the_containers_own(monkeypatch, fresh_s3_clients):
    # In Aspire the API container is told S3 lives at a container-network hostname. A presigned URL embeds
    # its host, so signing against that one hands the browser a link it cannot resolve.
    settings = get_settings()
    monkeypatch.setattr(settings, "s3_endpoint", "http://rustfs.dev.internal:9000")
    monkeypatch.setattr(settings, "s3_public_endpoint", "http://localhost:9010")

    url = urlsplit(storage.get_download_url("theseus-datasets", "pool/p/ab/abc.png"))

    assert (url.hostname, url.port) == ("localhost", 9010)
    assert url.path == "/theseus-datasets/pool/p/ab/abc.png"


def test_download_urls_use_sigv4_because_rustfs_rejects_the_legacy_signature(monkeypatch, fresh_s3_clients):
    # boto3 signs presigned URLs for a custom endpoint the legacy SigV2 way (AWSAccessKeyId/Signature/Expires)
    # unless told otherwise, and RustFS answers that with 403 SignatureDoesNotMatch.
    monkeypatch.setattr(get_settings(), "s3_public_endpoint", "http://localhost:9000")

    query = parse_qs(urlsplit(storage.get_download_url("b", "k")).query)

    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert "X-Amz-Signature" in query and "AWSAccessKeyId" not in query


def test_download_urls_fall_back_to_the_normal_endpoint_when_no_public_one_is_set(monkeypatch, fresh_s3_clients):
    settings = get_settings()
    monkeypatch.setattr(settings, "s3_endpoint", "http://localhost:9000")
    monkeypatch.setattr(settings, "s3_public_endpoint", None)

    assert urlsplit(storage.get_download_url("b", "k")).netloc == "localhost:9000"


def test_the_apps_own_s3_calls_keep_using_the_internal_endpoint(monkeypatch, fresh_s3_clients):
    settings = get_settings()
    monkeypatch.setattr(settings, "s3_endpoint", "http://rustfs.dev.internal:9000")
    monkeypatch.setattr(settings, "s3_public_endpoint", "http://localhost:9010")

    assert storage.s3().meta.endpoint_url == "http://rustfs.dev.internal:9000"
