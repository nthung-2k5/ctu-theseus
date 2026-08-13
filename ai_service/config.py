import os
from pathlib import Path


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


NATS_URI = _env("NATS_URI") or "nats://localhost:4222"
S3_ENDPOINT = _env("S3_ENDPOINT") or "http://localhost:9000"
S3_ACCESS_KEY = _env("S3_ACCESS_KEY") or "theseus"
S3_SECRET_KEY = _env("S3_SECRET_KEY") or "theseus-secret"
TEMP_DIR = Path(_env("TEMP_DIR") or "/tmp/theseus")

os.environ.setdefault("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY)
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", S3_SECRET_KEY)
os.environ.setdefault("AWS_ENDPOINT_URL", S3_ENDPOINT)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
