"""Settings for a remote worker: production needs its S3 settings, not the server's database or signing key."""

import pytest
from pydantic import ValidationError

from omnitrain.settings import Settings

S3 = {"S3_ENDPOINT": "http://s3.example", "S3_ACCESS_KEY": "k", "S3_SECRET_KEY": "s"}


@pytest.fixture
def production(monkeypatch):
    for name in ("CTU_OMNITRAIN_DB_URI", "JWT_SECRET", *S3):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    return monkeypatch


def test_a_production_worker_needs_only_s3(production):
    production.setenv("OMNITRAIN_ROLE", "worker")
    for name, value in S3.items():
        production.setenv(name, value)
    assert Settings(_env_file=None).role == "worker"


def test_a_production_worker_still_needs_s3(production):
    production.setenv("OMNITRAIN_ROLE", "worker")
    with pytest.raises(ValidationError, match="S3_ENDPOINT"):
        Settings(_env_file=None)


def test_a_production_server_still_needs_its_database_and_signing_key(production):
    for name, value in S3.items():
        production.setenv(name, value)
    with pytest.raises(ValidationError, match="CTU_OMNITRAIN_DB_URI, JWT_SECRET"):
        Settings(_env_file=None)


def test_the_remote_claim_timings_are_configurable_by_plain_env_vars(monkeypatch):
    monkeypatch.setenv("WORKER_ONLINE_SECONDS", "77")
    monkeypatch.setenv("REMOTE_CLAIM_GRACE_SECONDS", "9")
    s = Settings(_env_file=None)
    assert (s.worker_online_seconds, s.remote_claim_grace_seconds) == (77, 9)
