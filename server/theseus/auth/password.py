"""argon2id password hashing.

verify() costs 50-100 ms of CPU, so both operations run on a small dedicated pool. Otherwise a
login burst would stall the event loop that is also streaming SSE and feeding a GPU.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="argon2")


async def hash_password(password: str) -> str:
    return await asyncio.get_running_loop().run_in_executor(_pool, _hasher.hash, password)


def _verify(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


async def verify_password(password_hash: str, password: str) -> bool:
    return await asyncio.get_running_loop().run_in_executor(_pool, _verify, password_hash, password)


# A valid hash to verify against when the email is unknown, so login timing does not reveal
# which emails are registered.
DUMMY_HASH = _hasher.hash("theseus-dummy-password")
