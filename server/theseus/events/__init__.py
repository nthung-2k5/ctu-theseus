"""Run events: the bus, the single writer, and the SSE stream.

The bus and writer are process-wide singletons created by lifespan. Emitters anywhere (jobs,
routers, reapers) use get_event_writer(); the SSE endpoint uses get_event_bus().
"""

from theseus.events.bus import InProcessRunEventBus, RunEventBus
from theseus.events.writer import EventWriter

_bus: RunEventBus = InProcessRunEventBus()
_writer: EventWriter | None = None


def get_event_bus() -> RunEventBus:
    return _bus


def set_event_bus(bus: RunEventBus) -> None:
    global _bus
    _bus = bus


def get_event_writer() -> EventWriter:
    if _writer is None:
        raise RuntimeError("Event writer is not running (lifespan has not started it)")
    return _writer


def set_event_writer(writer: EventWriter | None) -> None:
    global _writer
    _writer = writer
