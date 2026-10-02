import uuid
from datetime import UTC, datetime, timedelta

import jwt

from theseus.settings import get_settings

ALGORITHM = "HS256"
ACCESS_TYPE = "access"


class InvalidToken(Exception):
    pass


def create_access_token(user_id: uuid.UUID) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    claims = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(seconds=s.access_token_ttl_seconds),
        "jti": uuid.uuid4().hex,
        "typ": ACCESS_TYPE,
    }
    return jwt.encode(claims, s.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> uuid.UUID:
    """Verify signature, expiry and type in-process. Zero DB reads: this is the stateless part."""
    try:
        claims = jwt.decode(
            token,
            get_settings().jwt_secret,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp", "iat", "typ"]},
        )
        if claims["typ"] != ACCESS_TYPE:
            raise InvalidToken("wrong token type")
        return uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError) as e:
        raise InvalidToken(str(e)) from e
