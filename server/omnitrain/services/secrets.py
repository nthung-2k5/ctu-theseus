"""Encrypting the third-party tokens users save, so a database dump alone never yields a usable credential.

Fernet (AES-128-CBC with an HMAC). The key is `OMNITRAIN_SECRETS_KEY` when set, else derived from the JWT secret
with HKDF under its own label, so the two uses never share raw key material. A value that cannot be decrypted
(the key changed, or the row was tampered with) is reported as `None`: the caller treats the token as gone and
asks the user to reconnect, never as a server error.
"""

import base64
import logging
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from omnitrain.settings import get_settings

logger = logging.getLogger(__name__)

# Pre-rename value, kept on purpose: it is part of the derived key, so changing it would make every saved token undecryptable.
_HKDF_INFO = b"theseus/secrets/v1"


@lru_cache
def _fernet(secrets_key: str | None, jwt_secret: str) -> Fernet:
    if secrets_key:
        return Fernet(secrets_key.encode())
    raw = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(jwt_secret.encode())
    return Fernet(base64.urlsafe_b64encode(raw))


def _cipher() -> Fernet:
    settings = get_settings()
    return _fernet(settings.secrets_key, settings.jwt_secret)


def encrypt(plaintext: str) -> str:
    return _cipher().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str | None:
    try:
        return _cipher().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        logger.warning("A stored secret could not be decrypted (the secrets key changed?)")
        return None
