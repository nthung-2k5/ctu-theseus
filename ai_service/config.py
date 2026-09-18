import os
from pathlib import Path


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


_IS_PRODUCTION = (_env("ENVIRONMENT", "NODE_ENV") or "").lower() == "production"


def _required(name: str, value: str | None, dev_fallback: str) -> str:
    """Dev-only fallback; in production a missing value fails at import time.

    Silently defaulting the S3 credentials surfaces as an opaque boto3 403
    deep inside a training run rather than at startup.
    """
    if value:
        return value
    if _IS_PRODUCTION:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return dev_fallback


NATS_URI = _required("NATS_URI", _env("NATS_URI"), "nats://localhost:4222")
S3_ENDPOINT = _required("S3_ENDPOINT", _env("S3_ENDPOINT"), "http://localhost:9000")
# Must match server/lib/config.ts — the two services previously defaulted to
# different credentials, so outside Aspire the worker authenticated with a
# key pair nothing issued.
S3_ACCESS_KEY = _required("S3_ACCESS_KEY", _env("S3_ACCESS_KEY"), "ctu-theseus")
S3_SECRET_KEY = _required("S3_SECRET_KEY", _env("S3_SECRET_KEY"), "ctu-theseus-secret")
TEMP_DIR = Path(_env("TEMP_DIR") or "/tmp/theseus")

# How many distinct runs' Ludwig models the inference model cache keeps
# loaded in memory at once (see services/model_cache.py). Bounds VRAM/RAM
# use — each loaded model stays resident until evicted LRU-style.
INFERENCE_MODEL_CACHE_SIZE = int(_env("INFERENCE_MODEL_CACHE_SIZE") or "2")

# Hard ceiling on one inference job (download + load + predict). Inference is
# dispatched asynchronously and polled (the gateway returns 202 + inferenceId,
# see server/routes/inference.ts), so this is NOT racing a request timeout —
# it only exists to stop a wedged job from occupying the worker forever. The
# four `modelType: 'llm'` tasks legitimately generate for minutes, so this is
# generous by design.
INFERENCE_TIMEOUT_SECONDS = int(_env("INFERENCE_TIMEOUT_SECONDS") or "600")

os.environ.setdefault("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY)
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", S3_SECRET_KEY)
os.environ.setdefault("AWS_ENDPOINT_URL", S3_ENDPOINT)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
