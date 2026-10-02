"""Replayable run-event stream for SSE (GET /api/runs/{run_id}/events).

Ordering is what makes reconnect lossless:

  1. subscribe to the live bus FIRST,
  2. then read the backlog from Postgres (seq > Last-Event-ID),
  3. emit the backlog,
  4. then emit live events, skipping any with seq <= the last one already sent.

Querying before subscribing would lose events committed in the gap between the two. Events
reach the bus only after commit, so anything the bus delivers is also in Postgres.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import sqlalchemy as sa

from theseus.db.base import get_sessionmaker
from theseus.db.models import RunEvent
from theseus.events.bus import CLOSED, RunEventBus

BACKLOG_PAGE = 500
# Idle period after which None is yielded so the caller can send an SSE comment. Proxies (YARP)
# drop idle connections, and a comment keeps the stream alive between sparse epoch events.
KEEPALIVE_SECONDS = 15.0


async def stream_run_events(
    bus: RunEventBus,
    run_id: uuid.UUID,
    after_seq: int = 0,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[dict[str, Any] | None]:
    """Yield {"seq", "kind", "payload", ...} dicts in seq order, or None as a keepalive tick.

    Ends when the subscriber falls too far behind (the client reconnects and heals from Postgres).
    """
    rid = str(run_id)
    sub = bus.subscribe(rid)
    try:
        last = after_seq
        while True:
            async with get_sessionmaker()() as session:
                rows = (
                    await session.execute(
                        sa.select(RunEvent.seq, RunEvent.kind, RunEvent.payload)
                        .where(RunEvent.run_id == run_id, RunEvent.seq > last)
                        .order_by(RunEvent.seq)
                        .limit(BACKLOG_PAGE)
                    )
                ).all()
            for seq, kind, payload in rows:
                last = seq
                yield {"seq": seq, "runId": rid, "kind": kind, "payload": payload}
            if len(rows) < BACKLOG_PAGE:
                break

        while True:
            try:
                item = await asyncio.wait_for(sub.queue.get(), keepalive_seconds)
            except TimeoutError:
                yield None
                continue
            if item is CLOSED:
                return
            if item["seq"] <= last:
                continue  # already delivered as part of the backlog
            last = item["seq"]
            yield item
    finally:
        bus.unsubscribe(sub)
