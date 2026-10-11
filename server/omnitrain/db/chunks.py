"""Splitting a long id list so one statement never carries too many bind parameters.

asyncpg refuses a statement with more than 32767 arguments, and `col IN (...)` sends one per value, so any
query that takes "every item in the draft" (a whole dataset can be hundreds of thousands) must be run in
chunks. The default leaves plenty of room for the statement's other parameters.
"""

from collections.abc import Iterator, Sequence

PARAM_CHUNK = 10_000


def chunked[T](values: Sequence[T], size: int = PARAM_CHUNK) -> Iterator[Sequence[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]
