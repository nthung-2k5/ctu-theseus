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
    # "worker" for a remote training worker (`python -m omnitrain.worker`), which has S3 but no database and
    # signs no tokens. The worker entrypoint sets it; nothing else should.
    role: str = Field(default="server", validation_alias="OMNITRAIN_ROLE")

    database_uri: str = Field(default=_DEV_DB, validation_alias="CTU_OMNITRAIN_DB_URI")

    s3_endpoint: str = Field(default="http://localhost:9000", validation_alias="S3_ENDPOINT")
    # Where the BROWSER can reach the object store, used only to sign the download URLs handed to
    # clients (`storage.get_download_url`). Unset means "same as s3_endpoint", which is right whenever
    # the API can reach S3 at an address the browser can too. It is NOT when the API runs in a container
    # and S3_ENDPOINT is a container-network hostname (Aspire injects http://rustfs.dev.internal:9000):
    # a presigned URL embeds that host, and the browser cannot resolve it.
    s3_public_endpoint: str | None = Field(default=None, validation_alias="S3_PUBLIC_ENDPOINT")
    s3_access_key: str = Field(default="ctu-omnitrain", validation_alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="ctu-omnitrain-secret", validation_alias="S3_SECRET_KEY")

    jwt_secret: str = Field(default=_DEV_JWT_SECRET, validation_alias="JWT_SECRET")
    # Encrypts third-party tokens users save (the Hugging Face integration) before they reach the database: a
    # Fernet key (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
    # Unset, a key is derived from JWT_SECRET, so existing deployments keep working; set it to rotate the
    # two independently. Changing either makes saved tokens unreadable, and users simply reconnect.
    secrets_key: str | None = Field(default=None, validation_alias="OMNITRAIN_SECRETS_KEY")
    access_token_ttl_seconds: int = 15 * 60
    refresh_token_ttl_seconds: int = 30 * 24 * 3600
    # None means Secure cookies in production only, so plain-http local dev still works.
    cookie_secure_override: bool | None = Field(default=None, validation_alias="COOKIE_SECURE")
    # Extra browser origins allowed to make cookie-authenticated writes (comma separated).
    allowed_origins: str = Field(default="", validation_alias="ALLOWED_ORIGINS")
    # Emails (comma separated) that are promoted to role=admin when they register or sign in. This is
    # the only way the first admin comes to exist; promotion never demotes, so removing an email
    # here does not strip an existing admin (do that from the admin users page).
    admin_emails: str = Field(default="", validation_alias="OMNITRAIN_ADMIN_EMAILS")
    # The administrator startup creates when the database has none (services/bootstrap_admin.py). Without a
    # password, development falls back to `dev_admin_password`; production seeds nothing.
    admin_email: str = Field(default="admin@omnitrain.local", validation_alias="OMNITRAIN_ADMIN_EMAIL")
    admin_password: str = Field(default="", validation_alias="OMNITRAIN_ADMIN_PASSWORD")
    dev_admin_password: str = "admin12345"

    temp_dir: Path = Field(default=Path("/tmp/omnitrain"), validation_alias="TEMP_DIR")
    inference_model_cache_size: int = Field(default=2, validation_alias="INFERENCE_MODEL_CACHE_SIZE")
    inference_timeout_seconds: int = Field(default=600, validation_alias="INFERENCE_TIMEOUT_SECONDS")
    # Largest bring-your-own model bundle, uploaded or fetched from the Hub. The browser uploads with a single
    # presigned PUT, which S3 caps at 5 GiB, so that is also the default.
    max_custom_model_bytes: int = Field(default=5 * 2**30, validation_alias="MAX_CUSTOM_MODEL_BYTES")

    port: int = Field(default=8000, validation_alias="PORT")

    # Job runner
    job_poll_interval_seconds: float = 3.0
    job_lease_seconds: int = 300
    inference_concurrency: int = 2
    # A running training run with no heartbeat or event for this long is presumed hung. The Ludwig callback
    # beats every few seconds while training, so this only needs to exceed the longest stretch without one
    # (dataset preprocessing).
    run_heartbeat_timeout_seconds: int = 900
    # Remote training workers (jobs/queue.py). A worker is "online" if it called the API within this many seconds;
    # workers poll every few seconds, so this leaves room for a missed poll or two.
    worker_online_seconds: int = 30
    # How long a queued run is held back for an idle online worker before this server trains it itself. It is what
    # stops a worker that is online but not claiming (a bug, a backend it lacks) from stalling the queue.
    remote_claim_grace_seconds: int = 20

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
    def admin_email_set(self) -> frozenset[str]:
        return frozenset(e.strip().lower() for e in self.admin_emails.split(",") if e.strip())

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
            "S3_ENDPOINT": "s3_endpoint",
            "S3_ACCESS_KEY": "s3_access_key",
            "S3_SECRET_KEY": "s3_secret_key",
        }
        if self.role != "worker":
            required |= {"CTU_OMNITRAIN_DB_URI": "database_uri", "JWT_SECRET": "jwt_secret"}
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
