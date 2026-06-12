"""
NATS + JetStream client for the AI worker service.

Provides connection management, stream provisioning, and publish/subscribe helpers.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import nats
import nats.errors
import nats.js.errors
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext
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

        # Provision Object Store for inference uploads
        try:
            await self._js.object_store("theseus-inferences")
            logger.info("Object Store 'theseus-inferences' exists.")
        except nats.js.errors.NotFoundError:
            await self._js.create_object_store(
                "theseus-inferences", description="Uploads for inference tasks"
            )
            logger.info("Object Store 'theseus-inferences' created.")

        logger.info("NATS JetStream connected and streams provisioned.")

    async def close(self) -> None:
        """Gracefully close the NATS connection."""
        if self._nc and self._nc.is_connected:
            await self._nc.drain()
            logger.info("NATS connection closed.")

    # ──────────────────────────────────────────────────────────────
    # Object Store
    # ──────────────────────────────────────────────────────────────

    async def get_upload(self, bucket: str, key: str) -> bytes | None:
        """Download an uploaded image from NATS Object Store."""
        os = await self.js.object_store(bucket)
        obj = await os.get(key)
        return obj.data

    async def download(self, bucket: str, key: str, downloadTo: Path) -> bool:
        """Download an uploaded image from NATS Object Store to a specific path."""
        os = await self.js.object_store(bucket)
        try:
            with open(downloadTo, "wb") as f:
                await os.get(key, writeinto=f)
            return True
        except Exception as e:
            logger.error(f"Failed to download {key} from NATS Object Store: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Publishing
    # ──────────────────────────────────────────────────────────────

    async def publish(self, subject: str, data: dict[str, Any]) -> None:
        """Publish a JSON message to a JetStream subject."""
        payload = json.dumps(data).encode()
        ack = await self.js.publish(subject, payload)
        logger.debug(f"Published to {subject} (stream={ack.stream}, seq={ack.seq})")

    async def publish_result(
        self, task_type: str, task_id: str, data: dict[str, Any]
    ) -> None:
        """Publish a result message."""
        await self.publish(f"theseus.results.{task_type}.{task_id}", data)

    async def publish_progress(self, run_id: str, data: dict[str, Any]) -> None:
        """Publish a training progress message."""
        await self.publish(f"theseus.progress.train.{run_id}", data)

    # ──────────────────────────────────────────────────────────────
    # Subscribing (pull-based consumers)
    # ──────────────────────────────────────────────────────────────

    # For Request - Reply
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Msg], Awaitable[None]],
    ) -> None:
        """Basic Request-Reply subscription."""
        await self.nc.subscribe(subject, cb=handler)
        logger.info(f"Subscribed to '{subject}' with request-reply")

    async def _consume_loop(
        self,
        consumer: JetStreamContext.PullSubscription,
        subject: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        retry_on_failure: bool = True,
        nak_delay: int = 10,
        fetch_timeout: int = 5,
    ) -> None:
        """
        Generic pull consumer loop.
        """
        while True:
            try:
                messages = await consumer.fetch(batch=1, timeout=fetch_timeout)
                for msg in messages:
                    try:
                        data = json.loads(msg.data.decode())
                        await handler(data)
                        await msg.ack()
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Malformed JSON in {subject}: {e}. Dropping message."
                        )
                        await msg.ack()  # Prevent poison pill infinite retries
                    except Exception as e:
                        logger.exception(f"Handler failed for {subject}: {e}")
                        if retry_on_failure:
                            await msg.nak(delay=nak_delay)
                        else:
                            await msg.ack()

            except nats.errors.TimeoutError:
                # No messages available, continue polling
                continue
            except asyncio.CancelledError:
                logger.info(
                    f"Consumer loop for '{subject}' cancelled. Shutting down cleanly."
                )
                break
            except Exception as e:
                # Check for disconnects (ensure self.nc is the correct reference)
                if getattr(self, "nc", None) is None or not self.nc.is_connected:
                    logger.error("NATS disconnected, stopping consumer loop.")
                    break
                logger.exception(f"Consumer polling error on '{subject}': {e}")
                await asyncio.sleep(1)

    async def subscribe_tasks(
        self,
        subject: str,
        durable_name: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Subscribe to task messages using a pull-based consumer.
        The handler receives the parsed JSON payload and should process it.
        Messages are acked after successful processing; nak'd on failure.
        """
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

        # Delegate to the generic loop (retries enabled)
        await self._consume_loop(
            consumer=consumer,
            subject=subject,
            handler=handler,
            retry_on_failure=True,
            nak_delay=10,
        )

    async def subscribe_commands(
        self,
        subject: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
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

        logger.info(f"Subscribed to '{subject}' for commands")

        # Delegate to the generic loop (retries disabled for commands)
        await self._consume_loop(
            consumer=consumer,
            subject=subject,
            handler=handler,
            retry_on_failure=False,
            fetch_timeout=1,
        )


# Module-level singleton
nats_service = NatsService()
