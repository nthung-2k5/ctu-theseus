"""`python -m omnitrain.worker`: run a training worker until interrupted."""

import logging
import os
import signal
import sys
import threading

from pydantic import ValidationError


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    # Before the settings are first read: a worker needs only S3, not the server's database or JWT secret.
    os.environ.setdefault("OMNITRAIN_ROLE", "worker")

    from omnitrain.settings import get_settings
    from omnitrain.worker.client import ApiClient
    from omnitrain.worker.config import WorkerConfig
    from omnitrain.worker.runner import describe, serve

    try:
        config = WorkerConfig()
        settings = get_settings()
    except ValidationError as e:
        print(f"The worker is not configured: {e}", file=sys.stderr)
        return 2

    shutdown = threading.Event()

    def _stop(signum, _frame) -> None:
        logging.getLogger("omnitrain.worker").info("Stopping (signal %s)", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break, and what a service wrapper sends to stop a process
        signal.signal(signal.SIGBREAK, _stop)

    info = describe()
    logging.getLogger("omnitrain.worker").info(
        "Worker %s serving %s: backends %s, GPU %s, S3 %s",
        info["hostname"],
        config.api_url,
        info["backends"] or "none",
        info["gpu"] or "none",
        settings.s3_endpoint,
    )
    client = ApiClient(config.api_url, config.token)
    try:
        return serve(config, client, shutdown)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
