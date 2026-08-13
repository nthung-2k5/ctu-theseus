import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
from opentelemetry import propagate, trace
from pydantic import BaseModel

from ai_service.config import NATS_URI

tracer = trace.get_tracer("theseus-worker")

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Stream definitions — mirrors src/lib/nats.ts.
#
#   THESEUS_TASKS    theseus.task.train.{runId}, theseus.task.export.{jobId}
#   THESEUS_EVENTS   theseus.event.run.{runId}.{kind}   (status|metric|log)
#   THESEUS_COMMANDS theseus.command.run.{runId}
# ──────────────────────────────────────────────────────────────────

# NOTE: nats-py's StreamConfig.max_age is in SECONDS.
STREAMS: list[StreamConfig] = [
    StreamConfig(
        name="THESEUS_TASKS",
        subjects=["theseus.task.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        max_age=24 * 3600,  # 24 hours in seconds
    ),
    StreamConfig(
        name="THESEUS_EVENTS",
        subjects=["theseus.event.>"],
        retention=RetentionPolicy.LIMITS,
        max_age=7 * 24 * 3600,  # 7 days
    ),
    StreamConfig(
        name="THESEUS_COMMANDS",
        subjects=["theseus.command.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        max_age=3600,  # 1 hour
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
        logger.info(f"Connecting to NATS at {NATS_URI}...")
        self._nc = await nats.connect(
            servers=NATS_URI,
            reconnect_time_wait=2,
            max_reconnect_attempts=-1,  # Retry forever
        )
        self._js = self._nc.jetstream()

        # Provision streams (idempotent — updates if already exist)
        jsm = self._nc.jsm()
        for stream_config in STREAMS:
            try:
                if stream_config.subjects:
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
        """Publish a JSON message to a JetStream subject. Injects the current
        span's W3C traceparent into message headers so a consumer on the
        other side of the NATS hop can continue the same trace."""
        payload = json.dumps(data).encode()
        headers: dict[str, str] = {}
        propagate.inject(headers)
        ack = await self.js.publish(subject, payload, headers=headers)
        logger.debug(f"Published to {subject} (stream={ack.stream}, seq={ack.seq})")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def publish_event(self, run_id: str, kind: str, data: dict[str, Any]) -> None:
        """
        Publish a run event (RunEventSchema in src/lib/schema.ts — status,
        metric, or log) to the run's subject on THESEUS_EVENTS. The
        JetStream sequence number becomes the SSE replay cursor, so every
        event a run emits — status, metrics, logs — goes through here.
        """
        await self.publish(
            f"theseus.event.run.{run_id}.{kind}",
            {"kind": kind, "runId": run_id, "ts": self._now(), **data},
        )

    async def publish_status(
        self,
        run_id: str,
        status: Literal["queued", "running", "succeeded", "failed", "canceled"],
        message: str | None = None,
    ) -> None:
        """Publish a status-kind run event."""
        data: dict[str, Any] = {"status": status}
        if message is not None:
            data["message"] = message
        await self.publish_event(run_id, "status", data)

    async def publish_metric(
        self, run_id: str, epoch: int, split: str, metrics: dict[str, float]
    ) -> None:
        """Publish a metric-kind run event (one or more named metrics for one epoch/split)."""
        await self.publish_event(
            run_id, "metric", {"epoch": epoch, "split": split, "metrics": metrics}
        )

    async def publish_log(
        self, run_id: str, line: str, level: Literal["info", "warn", "error"] = "info"
    ) -> None:
        """Publish a log-kind run event."""
        await self.publish_event(run_id, "log", {"level": level, "line": line})

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

    async def _consume_loop[T: BaseModel](
        self,
        consumer: JetStreamContext.PullSubscription,
        subject: str,
        handler: Callable[[T], Awaitable[None]],
        message_type: type[T],
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
                        parent_ctx = propagate.extract(msg.headers or {})
                        data = message_type.model_validate_json(msg.data.decode())
                        with tracer.start_as_current_span(
                            f"nats.consume {subject}", context=parent_ctx
                        ):
                            await handler(data)
                        await msg.ack()
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Malformed JSON in {subject}: {e}. Dropping message."
                        )
                        await msg.ack()  # Prevent poison pill infinite retries
                    except Exception:
                        logger.exception(f"Handler failed for {subject}")
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
            except Exception:
                # Check for disconnects (ensure self.nc is the correct reference)
                if getattr(self, "nc", None) is None or not self.nc.is_connected:
                    logger.error("NATS disconnected, stopping consumer loop.")
                    break
                logger.exception(f"Consumer polling error on '{subject}'")
                await asyncio.sleep(1)

    async def subscribe_tasks[T: BaseModel](
        self,
        subject: str,
        durable_name: str,
        handler: Callable[[T], Awaitable[None]],
        message_type: type[T],
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
            message_type=message_type,
            retry_on_failure=True,
            nak_delay=10,
        )

    async def subscribe_commands[T: BaseModel](
        self,
        subject: str,
        handler: Callable[[T], Awaitable[None]],
        message_type: type[T],
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
            message_type=message_type,
            retry_on_failure=False,
            fetch_timeout=1,
        )


# Module-level singleton
nats_service = NatsService()
