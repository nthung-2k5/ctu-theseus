"""Process startup/shutdown. Job dispatchers, the event writer and reapers are added here as they land.

Schema migrations are NOT run here: `alembic upgrade head` runs in the container entrypoint
before uvicorn starts, because a slow migration inside lifespan would fail the health check.
"""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from theseus.db.base import dispose_engine
from theseus.events import EventWriter, get_event_bus, set_event_writer, set_log_handler
from theseus.events.log_handler import install_run_log_handler, uninstall_run_log_handler
from theseus.jobs.reapers import reaper_loop
from theseus.jobs.recovery import recover_on_startup
from theseus.services import storage
from theseus.settings import get_settings
from theseus.telemetry import init_telemetry

logger = logging.getLogger(__name__)


def _workers_arg(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--workers" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--workers="):
            return arg.split("=", 1)[1]
    return None


def assert_single_process(argv: list[str] | None = None) -> None:
    """The service must run as exactly one process.

    SSE fanout, the GPU lane, the model cache, the rate limiter and the abort registry all live
    in process memory, and uvicorn workers fork (a forked child cannot reuse a CUDA context).
    Running more than one worker, or with --reload, breaks all of these silently.
    """
    argv = sys.argv if argv is None else argv
    problems = []
    concurrency = os.environ.get("WEB_CONCURRENCY", "1")
    if concurrency not in ("", "1"):
        problems.append(f"WEB_CONCURRENCY={concurrency}")
    workers = _workers_arg(argv)
    if workers not in (None, "1"):
        problems.append(f"--workers {workers}")
    if "--reload" in argv:
        problems.append("--reload")
    if problems:
        raise RuntimeError("Theseus must run as a single process, but found: " + ", ".join(problems))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    assert_single_process()
    get_settings()  # fail fast on missing production config
    init_telemetry(app)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, storage.ensure_buckets)

    # 1. The single event writer and the contextvar-scoped training log handler.
    writer = EventWriter(get_event_bus())
    await writer.start()
    set_event_writer(writer)
    log_handler = install_run_log_handler(writer)
    set_log_handler(log_handler)

    # 2. Settle whatever a previous process left in flight, BEFORE anything new is claimed.
    await recover_on_startup()

    # 3. Job lanes and housekeeping. Imported here because train/export pull in torch and Ludwig,
    #    and importing the app (for the OpenAPI export) must stay cheap.
    from theseus.jobs.dispatcher import Dispatcher, build_default_lanes, set_dispatcher

    dispatcher = Dispatcher(build_default_lanes())
    set_dispatcher(dispatcher)
    await dispatcher.start()
    reaper = asyncio.create_task(reaper_loop(), name="reaper")
    logger.info("Theseus API started.")
    try:
        yield
    finally:
        logger.info("Theseus API shutting down.")
        reaper.cancel()
        await asyncio.gather(reaper, return_exceptions=True)
        await dispatcher.stop()
        set_dispatcher(None)
        await writer.stop()
        uninstall_run_log_handler(log_handler)
        set_log_handler(None)
        set_event_writer(None)
        await dispose_engine()
