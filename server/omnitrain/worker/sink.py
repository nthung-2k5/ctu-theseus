"""The `run_core.ProgressSink` for a run training on a worker: metrics and heartbeats go to the API over HTTP.

The training thread only ever appends to an in-memory buffer or sets a flag, so a slow or unreachable API never slows
an epoch down. A second thread sends the buffer every few seconds. A batch is dropped from the buffer only once the API
accepted it, so an outage loses nothing, and a re-sent metric is harmless (the API upserts by epoch, split and name).
The same call carries the answer to "should I stop?": cancel, or a run the server already ended.
"""

import logging
import threading
import time
from typing import Any

from omnitrain.jobs.abort import TrainingAborted
from omnitrain.worker.client import ApiClient, is_transient

logger = logging.getLogger("omnitrain.worker")

MAX_BATCH = 500  # the API's cap per call
FINAL_FLUSH_ATTEMPTS = 10
FINAL_FLUSH_RETRY_SECONDS = 2.0


class HttpSink:
    def __init__(self, client: ApiClient, run_id: str, shutdown: threading.Event, flush_interval: float = 5.0) -> None:
        self._client = client
        self._run_id = run_id
        self._shutdown = shutdown
        self._interval = flush_interval
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._alive = False
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"flush-{run_id}", daemon=True)
        # True once the server says the run is over (canceled, or failed by its reaper): nothing is left to report.
        self.terminal = False

    # -- ProgressSink (called from the training thread) ----------------------------------------

    def heartbeat(self) -> None:
        self._alive = True

    def check_abort(self) -> None:
        if self._shutdown.is_set():
            raise TrainingAborted("The worker is shutting down")
        if self._cancel.is_set():
            raise TrainingAborted(f"Training aborted for run {self._run_id}")

    def report(self, epoch: int, split: str, metrics: dict[str, float]) -> None:
        entry = {"epoch": epoch, "split": split, "metrics": {k: float(v) for k, v in metrics.items()}}
        with self._lock:
            self._buffer.append(entry)

    # -- Sending ---------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the flusher and deliver what is still buffered, retrying a few times if the API is unreachable."""
        self._done.set()
        self._thread.join()
        for _ in range(FINAL_FLUSH_ATTEMPTS):
            if not self._pending():
                break
            if not self._flush():
                time.sleep(FINAL_FLUSH_RETRY_SECONDS)

    def _pending(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    def _loop(self) -> None:
        while not self._done.wait(self._interval):
            self._flush()

    def _flush(self) -> bool:
        """Send one batch. Returns whether the API accepted it."""
        with self._lock:
            batch = self._buffer[:MAX_BATCH]
        alive, self._alive = self._alive, False
        try:
            answer = self._client.events(self._run_id, batch, alive)
        except Exception as e:
            self._alive = self._alive or alive
            if is_transient(e):
                logger.warning("Could not reach the API to report progress for run %s: %s", self._run_id, e)
            else:
                logger.error("The API refused progress for run %s: %s", self._run_id, e)
            return False
        with self._lock:
            del self._buffer[: len(batch)]
        if answer.get("terminal"):
            self.terminal = True
        if answer.get("cancel") or answer.get("terminal"):
            self._cancel.set()
        return True
