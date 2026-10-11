"""Live fan-out of committed run events to SSE subscribers.

RunEventBus is an interface on purpose: with one process an in-memory implementation is all
that is needed, and LISTEN/NOTIFY would be a small drop-in replacement if the job runner is
ever split into its own process. Events reach the bus only AFTER their database transaction
commits, so a subscriber can never see an event that a replay would not also return.
"""

import asyncio
from typing import Any, Protocol

# A subscriber that falls this far behind is closed instead of being fed a silently lossy
# stream. EventSource reconnects with Last-Event-ID and heals from Postgres, so disconnecting
# is a safe backpressure strategy.
SUBSCRIBER_QUEUE_SIZE = 1000

# Put on a subscriber queue to tell its consumer the subscription was closed.
CLOSED = object()


class Subscription:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.closed = False

    def close(self) -> None:
        """Drop anything buffered and wake the consumer with the CLOSED sentinel."""
        self.closed = True
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(CLOSED)


class RunEventBus(Protocol):
    def subscribe(self, run_id: str) -> Subscription: ...
    def unsubscribe(self, sub: Subscription) -> None: ...
    def publish(self, run_id: str, event: dict[str, Any]) -> None: ...


class InProcessRunEventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[Subscription]] = {}

    def subscribe(self, run_id: str) -> Subscription:
        sub = Subscription(run_id)
        self._subs.setdefault(run_id, set()).add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        subs = self._subs.get(sub.run_id)
        if subs is None:
            return
        subs.discard(sub)
        if not subs:
            del self._subs[sub.run_id]

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        for sub in list(self._subs.get(run_id, ())):
            if sub.closed:
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                sub.close()
                self.unsubscribe(sub)

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subs.get(run_id, ()))
