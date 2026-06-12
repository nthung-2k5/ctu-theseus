"""
Theseus AI Microservice -- Main entry point.
"""

import asyncio
import logging

from aiohttp import web

from ai_service.tasks import close_task_workers, run_task_workers

logger = logging.getLogger(__name__)


async def healthcheck(_request):
    """Health check endpoint for Docker / load-balancer probes."""
    return web.json_response({"status": "healthy"})


async def task_worker_context(_app):
    """
    Context manager to tie background tasks to the app's lifecycle.
    This ensures NATS workers don't block the web server and shut down cleanly.
    """
    logger.info("Starting AI Microservice...")

    worker_task = asyncio.create_task(run_task_workers())

    # Yield control back to aiohttp so it can start the web server
    yield

    # Shutdown sequence begins here (triggered by SIGINT/SIGTERM)
    logger.info("Shutting down AI Microservice...")

    # Close the NATS connections first to stop accepting new work
    await close_task_workers()

    # Cancel the infinite 'while True' polling loops
    worker_task.cancel()
    try:
        # Await the cancelled task to ensure it exits cleanly
        await worker_task
    except asyncio.CancelledError:
        logger.info("AI Microservice cleanly stopped.")


def main():
    # Set up the aiohttp app
    app = web.Application()
    app.router.add_get("/health", healthcheck)

    app.cleanup_ctx.append(task_worker_context)

    # web.run_app automatically handles the asyncio event loop,
    # OS signals (SIGINT/SIGTERM), and graceful shutdown.
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    main()
