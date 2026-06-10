"""
NATS JetStream client for the AI worker service.

Provides connection management, stream provisioning, and publish/subscribe
helpers. Replaces Redis/Celery as the task queue and result transport.
"""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Coroutine

import nats
from nats.aio.client import Client as NatsClient
from nats.js import JetStreamContext
import nats.js.errors
import nats.errors
from nats.js.api import (
    ConsumerConfig,
    DeliverPolicy,
    RetentionPolicy,
    StreamConfig,
)

logger = logging.getLogger(__name__)

NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")

# ──────────────────────────────────────────────────────────────────
# Stream definitions
# ──────────────────────────────────────────────────────────────────

STREAMS: list[StreamConfig] = [
    StreamConfig(
        name="TASKS",
        subjects=["theseus.tasks.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        max_age=24 * 3600 * 1_000_000_000,  # 24 hours in nanoseconds
    ),
    StreamConfig(
        name="RESULTS",
        subjects=["theseus.results.>"],
        retention=RetentionPolicy.LIMITS,
        max_age=7 * 24 * 3600 * 1_000_000_000,  # 7 days
    ),
    StreamConfig(
        name="PROGRESS",
        subjects=["theseus.progress.>"],
        retention=RetentionPolicy.LIMITS,
        max_age=3600 * 1_000_000_000,  # 1 hour
    ),
    StreamConfig(
        name="COMMANDS",
        subjects=["theseus.commands.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        max_age=3600 * 1_000_000_000,  # 1 hour
    ),
]


class NatsService:
    """Manages the NATS connection and JetStream context for the AI worker."""

    def __init__(self):
        self._nc: NatsClient | None = None
        self._js: JetStreamContext | None = None

    @property
    def nc(self) -> NatsClient:
        assert self._nc is not None, "NATS not connected. Call connect() first."
        return self._nc

    @property
    def js(self) -> JetStreamContext:
        assert self._js is not None, "JetStream not initialized. Call connect() first."
        return self._js

    async def connect(self) -> None:
        """Connect to NATS and provision JetStream streams."""
        logger.info(f"Connecting to NATS at {NATS_URL}...")
        self._nc = await nats.connect(
            servers=NATS_URL,
            reconnect_time_wait=2,
            max_reconnect_attempts=-1,  # Retry forever
        )
        self._js = self._nc.jetstream()

        # Provision streams (idempotent — updates if already exist)
        jsm = self._nc.jsm()
        for stream_config in STREAMS:
            try:
                # pyrefly: ignore [unsupported-operation]
                await jsm.find_stream_name_by_subject(stream_config.subjects[0])
                # Stream exists, update it
                await jsm.update_stream(stream_config)
                logger.info(f"Stream '{stream_config.name}' updated.")
            except nats.js.errors.NotFoundError:
                await jsm.add_stream(stream_config)
                logger.info(f"Stream '{stream_config.name}' created.")

        logger.info("NATS JetStream connected and streams provisioned.")

    async def close(self) -> None:
        """Gracefully close the NATS connection."""
        if self._nc and self._nc.is_connected:
            await self._nc.drain()
            logger.info("NATS connection closed.")

    # ──────────────────────────────────────────────────────────────
    # Publishing
    # ──────────────────────────────────────────────────────────────

    async def publish(self, subject: str, data: dict[str, Any]) -> None:
        """Publish a JSON message to a JetStream subject."""
        payload = json.dumps(data).encode()
        ack = await self.js.publish(subject, payload)
        logger.debug(f"Published to {subject} (stream={ack.stream}, seq={ack.seq})")

    async def publish_result(self, task_type: str, task_id: str, data: dict[str, Any]) -> None:
        """Publish a result message."""
        await self.publish(f"theseus.results.{task_type}.{task_id}", data)

    async def publish_progress(self, run_id: str, data: dict[str, Any]) -> None:
        """Publish a training progress message."""
        await self.publish(f"theseus.progress.train.{run_id}", data)

    # ──────────────────────────────────────────────────────────────
    # Subscribing (pull-based consumers)
    # ──────────────────────────────────────────────────────────────

    async def subscribe_tasks(
        self,
        subject: str,
        durable_name: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Subscribe to task messages using a pull-based consumer.
        The handler receives the parsed JSON payload and should process it.
        Messages are acked after successful processing; nak'd on failure.
        """
        # Create or bind to a durable pull consumer
        consumer = await self.js.pull_subscribe(
            subject,
            durable=durable_name,
            config=ConsumerConfig(
                durable_name=durable_name,
                deliver_policy=DeliverPolicy.ALL,
                ack_wait=3600,  # 1 hour ack timeout (training can be long)
                max_deliver=3,  # Retry up to 3 times on failure
            ),
        )

        logger.info(f"Subscribed to '{subject}' as consumer '{durable_name}'")

        while True:
            try:
                messages = await consumer.fetch(batch=1, timeout=5)
                for msg in messages:
                    try:
                        data = json.loads(msg.data.decode())
                        await handler(data)
                        await msg.ack()
                    except Exception as e:
                        logger.exception(f"Task handler failed for {subject}: {e}")
                        await msg.nak(delay=10)  # Retry after 10 seconds
            except nats.errors.TimeoutError:
                # No messages available, continue polling
                continue
            except Exception as e:
                if not self._nc or not self._nc.is_connected:
                    logger.error("NATS disconnected, stopping consumer loop.")
                    break
                logger.exception(f"Consumer error: {e}")
                await asyncio.sleep(1)

    async def subscribe_commands(
        self,
        subject: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to command messages (stop/abort)."""
        consumer = await self.js.pull_subscribe(
            subject,
            durable="commands-worker",
            config=ConsumerConfig(
                durable_name="commands-worker",
                deliver_policy=DeliverPolicy.NEW,
                ack_wait=30,
            ),
        )

        while True:
            try:
                messages = await consumer.fetch(batch=1, timeout=1)
                for msg in messages:
                    try:
                        data = json.loads(msg.data.decode())
                        await handler(data)
                        await msg.ack()
                    except Exception as e:
                        logger.exception(f"Command handler failed: {e}")
                        await msg.ack()  # Don't retry commands
            except nats.errors.TimeoutError:
                continue
            except Exception:
                if not self._nc or not self._nc.is_connected:
                    break
                await asyncio.sleep(1)


# Module-level singleton
nats_service = NatsService()
