"""In-process fixed-window rate limiter for the hosted prediction API.

Per-process state is correct because the service is pinned to a single process (workers=1 is
asserted at startup). Keys must never be raw secrets; use the API key hash.
"""

import time
from dataclasses import dataclass


@dataclass
class _Window:
    count: int
    start: float


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


_windows: dict[str, _Window] = {}


def check(key: str, max_requests: int, window_seconds: float, now: float | None = None) -> RateLimitResult:
    """Count one request against key and report whether it is within max_requests per window."""
    now = time.monotonic() if now is None else now
    w = _windows.get(key)
    if w is None or now - w.start >= window_seconds:
        w = _Window(count=0, start=now)
        _windows[key] = w
    w.count += 1
    retry_after = max(1, int(w.start + window_seconds - now) + 1)
    return RateLimitResult(
        allowed=w.count <= max_requests,
        limit=max_requests,
        remaining=max(0, max_requests - w.count),
        retry_after=retry_after,
    )


def sweep(window_seconds: float, now: float | None = None) -> int:
    """Drop expired windows so the map does not grow for the life of a long-running process."""
    now = time.monotonic() if now is None else now
    stale = [k for k, w in _windows.items() if now - w.start >= window_seconds]
    for k in stale:
        del _windows[k]
    return len(stale)


def reset() -> None:
    _windows.clear()
