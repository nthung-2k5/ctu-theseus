"""API key generation/hashing for the hosted prediction API (/api/v1).

Only a sha256 hash is ever persisted; the raw key is returned once at creation, like a password.
"""

import hashlib
import secrets

KEY_PREFIX = "thsk_"
RAW_KEY_BYTES = 24
# Enough of the raw key to tell keys apart in a list, not enough to be useful to an attacker.
DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 6


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_hash, key_prefix)."""
    raw = KEY_PREFIX + secrets.token_hex(RAW_KEY_BYTES)
    return raw, hash_api_key(raw), raw[:DISPLAY_PREFIX_LENGTH]
