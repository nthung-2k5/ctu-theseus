from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseSettings):
    """What the worker needs beyond the S3 settings it shares with the server (`omnitrain.settings`)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # The server's address as this machine reaches it, e.g. https://omnitrain.example (no /api suffix).
    api_url: str = Field(validation_alias="OMNITRAIN_API_URL")
    # The `thwk_...` token shown once when the worker is registered in Admin -> Workers.
    token: str = Field(validation_alias="OMNITRAIN_WORKER_TOKEN")
    # How often an idle worker asks for work. It must stay well under the server's `worker_online_seconds`, which is
    # how long the server believes a worker is there after hearing from it.
    poll_interval_seconds: float = Field(default=3.0, gt=0, validation_alias="OMNITRAIN_WORKER_POLL_SECONDS")
    # How often a training worker sends buffered metrics and its heartbeat.
    flush_interval_seconds: float = Field(default=5.0, gt=0, validation_alias="OMNITRAIN_WORKER_FLUSH_SECONDS")
