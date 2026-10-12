"""Bearer tokens for remote training workers (/api/worker).

The same scheme as the hosted-API keys in api_keys.py, with its own prefix so a token pasted into the wrong place is
recognisable: only a sha256 hash is persisted, and the raw token is shown once, when the worker is registered.
"""

import hashlib
import secrets

TOKEN_PREFIX = "thwk_"
RAW_TOKEN_BYTES = 24


def hash_worker_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def generate_worker_token() -> tuple[str, str]:
    """Returns (raw_token, token_hash)."""
    raw = TOKEN_PREFIX + secrets.token_hex(RAW_TOKEN_BYTES)
    return raw, hash_worker_token(raw)
