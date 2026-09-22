"""Stream training log lines to the browser and to a per-run log file.

The legacy worker attached a handler to the ROOT logger, which was safe only because that
process had no HTTP traffic. Merged into the API process the root logger also carries uvicorn
access logs, SQLAlchemy and botocore, all of which would be persisted as "training logs" of
whatever run happened to be active, and a log line emitted while writing a log event would
recurse.

Instead ONE handler is installed at startup on the `theseus.jobs` namespace plus every installed
trainer backend's own logger namespace(s) (`TrainerBackend.log_namespaces`, e.g. Ludwig's `ludwig`),
and it acts on a record only when the `current_run_id` ContextVar is set. Contextvars propagate
through asyncio tasks and (via contextvars.copy_context) into the training thread, the same
mechanism the OpenTelemetry span already relies on. There is no addHandler/removeHandler churn
per run and no cross-talk between concurrent runs.
"""

import logging
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO

from theseus.events.writer import EventWriter

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# Always installed. A trainer backend contributes its own additional namespace(s) (see
# TrainerBackend.log_namespaces) via `install_run_log_handler`'s `extra_namespaces`.
BASE_LOGGER_NAMESPACES = ("theseus.jobs",)

_LEVELS = {
    logging.DEBUG: "info",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}

# Live stream throttle (token bucket). The complete, unthrottled log always goes to the file.
RATE_PER_SECOND = 20.0
BURST = 200.0


class _Bucket:
    def __init__(self) -> None:
        self.tokens = BURST
        self.updated = time.monotonic()
        self.dropped = 0

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(BURST, self.tokens + (now - self.updated) * RATE_PER_SECOND)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        self.dropped += 1
        return False


class RunLogHandler(logging.Handler):
    def __init__(self, writer: EventWriter) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        self._writer = writer
        self._lock = threading.Lock()
        self._files: dict[str, TextIO] = {}
        self._buckets: dict[str, _Bucket] = {}
        # Set by install_run_log_handler right after construction; kept here so
        # uninstall_run_log_handler knows what to remove itself from.
        self.namespaces: tuple[str, ...] = ()

    def attach(self, run_id: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._files[run_id] = open(path, "a", encoding="utf-8")  # noqa: SIM115 (closed in detach)
            self._buckets[run_id] = _Bucket()

    def detach(self, run_id: str) -> None:
        with self._lock:
            f = self._files.pop(run_id, None)
            self._buckets.pop(run_id, None)
        if f is not None:
            f.close()

    def emit(self, record: logging.LogRecord) -> None:
        run_id = current_run_id.get()
        if run_id is None:
            return
        try:
            line = self.format(record)
        except Exception:
            return
        level = _LEVELS.get(record.levelno, "info")
        note: str | None = None
        with self._lock:
            f = self._files.get(run_id)
            if f is not None:
                f.write(line + "\n")
                f.flush()
            bucket = self._buckets.get(run_id)
            allowed = bucket.take() if bucket is not None else True
            if allowed and bucket is not None and bucket.dropped:
                note = f"{bucket.dropped} log lines omitted from the live stream (full log is in the run logs download)"
                bucket.dropped = 0
        if note:
            self._writer.log(run_id, "info", note)
        if allowed:
            self._writer.log(run_id, level, line)


def install_run_log_handler(writer: EventWriter, extra_namespaces: tuple[str, ...] = ()) -> RunLogHandler:
    handler = RunLogHandler(writer)
    handler.namespaces = BASE_LOGGER_NAMESPACES + extra_namespaces
    for name in handler.namespaces:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
    return handler


def uninstall_run_log_handler(handler: RunLogHandler) -> None:
    for name in handler.namespaces:
        logging.getLogger(name).removeHandler(handler)
