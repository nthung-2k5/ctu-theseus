"""Environment-driven settings.

Dev-only fallbacks exist for local runs; in production (ENVIRONMENT=production) a missing
required value fails at startup instead of silently pointing at a local Postgres/S3 or
signing tokens with a known secret.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_DB = "postgres://theseus:theseus@localhost:5432/theseus"
_DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    environment: str = Field(default="development", validation_alias="ENVIRONMENT")

    database_uri: str = Field(default=_DEV_DB, validation_alias="CTU_THESEUS_DB_URI")

    s3_endpoint: str = Field(default="http://localhost:9000", validation_alias="S3_ENDPOINT")
    s3_access_key: str = Field(default="ctu-theseus", validation_alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="ctu-theseus-secret", validation_alias="S3_SECRET_KEY")

    jwt_secret: str = Field(default=_DEV_JWT_SECRET, validation_alias="JWT_SECRET")
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_seconds: int = 30 * 24 * 3600
    # None means Secure cookies in production only, so plain-http local dev still works.
    cookie_secure_override: bool | None = Field(default=None, validation_alias="COOKIE_SECURE")
    # Extra browser origins allowed to make cookie-authenticated writes (comma separated).
    allowed_origins: str = Field(default="", validation_alias="ALLOWED_ORIGINS")

    temp_dir: Path = Field(default=Path("/tmp/theseus"), validation_alias="TEMP_DIR")
    inference_model_cache_size: int = Field(default=2, validation_alias="INFERENCE_MODEL_CACHE_SIZE")
    inference_timeout_seconds: int = Field(default=600, validation_alias="INFERENCE_TIMEOUT_SECONDS")

    port: int = Field(default=8000, validation_alias="PORT")

    # Job runner
    job_poll_interval_seconds: float = 3.0
    job_lease_seconds: int = 300
    inference_concurrency: int = 2
    # A running training run with no event for this long is presumed hung. Log lines count as
    # events, so this only needs to exceed the longest silent stretch (e.g. dataset preprocessing).
    run_heartbeat_timeout_seconds: int = 900
    sync_predict_max_wait_seconds: float = 25.0
    # Persisted log rows per run in run_events; the full log always goes to S3 regardless.
    run_log_max_rows: int = 20_000

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def cookie_secure(self) -> bool:
        if self.cookie_secure_override is not None:
            return self.cookie_secure_override
        return self.is_production

    @property
    def allowed_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def async_database_url(self) -> str:
        """SQLAlchemy needs an explicit asyncpg driver in the URL scheme."""
        url = self.database_uri
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix) :]
        return url

    @model_validator(mode="after")
    def _require_in_production(self) -> "Settings":
        if not self.is_production:
            return self
        required = {
            "CTU_THESEUS_DB_URI": "database_uri",
            "S3_ENDPOINT": "s3_endpoint",
            "S3_ACCESS_KEY": "s3_access_key",
            "S3_SECRET_KEY": "s3_secret_key",
            "JWT_SECRET": "jwt_secret",
        }
        missing = [env for env, field in required.items() if field not in self.model_fields_set]
        if missing:
            raise ValueError("Missing required environment variables in production: " + ", ".join(missing))
        return self


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # s3fs (used by Ludwig for s3:// paths) reads the standard AWS_* variables.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", s.s3_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", s.s3_secret_key)
    os.environ.setdefault("AWS_ENDPOINT_URL", s.s3_endpoint)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    return s
