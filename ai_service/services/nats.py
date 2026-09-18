import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import nats
import nats.errors
import nats.js.errors
from config import NATS_URI
from nats.aio.client import Client as NatsClient
from nats.aio.errors import NatsError
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import (
    ConsumerConfig,
    DeliverPolicy,
    RetentionPolicy,
    StreamConfig,
)
from opentelemetry import propagate, trace
from pydantic import BaseModel, ValidationError

from schema import subjects as subj
from services.metrics import nats_task_count, nats_task_duration

tracer = trace.get_tracer("theseus-worker")

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Stream definitions — generated from server/lib/subjects.ts (see
# schema/subjects.py) into schema.subjects.STREAMS; this is the single
# source of truth, mirrored on the gateway side by lib/nats.ts.
# ──────────────────────────────────────────────────────────────────

_RETENTION_MAP = {"workqueue": RetentionPolicy.WORK_QUEUE, "limits": RetentionPolicy.LIMITS}


def _to_stream_config(d: subj.StreamDef) -> StreamConfig:
    kwargs: dict[str, Any] = {
        "name": d["name"],
        "subjects": d["subjects"],
        "retention": _RETENTION_MAP[d["retention"]],
        "max_age": d["max_age_seconds"],  # nats-py's StreamConfig.max_age is in SECONDS
    }
    if "max_msgs_per_subject" in d:
        kwargs["max_msgs_per_subject"] = d["max_msgs_per_subject"]
    return StreamConfig(**kwargs)


STREAMS: list[StreamConfig] = [_to_stream_config(d) for d in subj.STREAMS]


# How long a task consumer waits for an ack before redelivering. This is a
# renewable lease, not a hard deadline: `_ack_keepalive` below pings
# `in_progress()` while a handler runs, so a legitimately long training run
# never trips it. Kept close to the gateway's HEARTBEAT_STALE_MS
# (server/lib/microservice.ts) so a dead worker is detected by both sides at
# roughly the same time instead of 55 minutes apart.
TASK_ACK_WAIT_SECONDS = 300
_ACK_KEEPALIVE_INTERVAL = 30


@contextlib.asynccontextmanager
async def _ack_keepalive(msg: Msg, interval: float = _ACK_KEEPALIVE_INTERVAL) -> AsyncIterator[None]:
    """Renew a message's ack lease while its handler runs.

    Without this, `ack_wait` is a hard ceiling on handler duration: a training
    run that outlives it is redelivered and — because the consume loop is
    serialized — retrained in full the moment the first attempt finishes,
    producing a second `succeeded` event for the same run.
    """

    async def beat() -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await msg.in_progress()
            except Exception:
                logger.debug("ack keepalive failed (message may already be acked)", exc_info=True)

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


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
            f = await asyncio.to_thread(lambda: downloadTo.open("wb"))
            try:
                await os.get(key, writeinto=f)
            finally:
                await asyncio.to_thread(f.close)

            return True
        except (NatsError, OSError) as e:
            logger.error(f"Failed to download {key} from NATS Object Store: {e}")
            return False

    async def delete_upload(self, bucket: str, key: str) -> None:
        """Remove an object from the store once it's been consumed —
        best-effort; a failure here just means the sweep reaper
        (server/lib/microservice.ts) picks it up later."""
        try:
            os = await self.js.object_store(bucket)
            await os.delete(key)
        except (NatsError, nats.js.errors.NotFoundError) as e:
            logger.warning(f"Failed to delete {key} from NATS Object Store: {e}")

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
            subj.run_event(run_id, kind),
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

    async def publish_export_event(
        self,
        run_id: str,
        job_id: str,
        status: Literal["success", "failed"],
        format: Literal["onnx", "torchscript"] | None = None,
        export_key: str | None = None,
        error: str | None = None,
    ) -> None:
        """Publish an export-kind run event (RunEventSchema's fourth member —
        see server/lib/schema.ts). The gateway's run-events consumer uses
        this to move an `exports` row from 'converting' to 'assembling'."""
        data: dict[str, Any] = {"jobId": job_id, "status": status}
        if format is not None:
            data["format"] = format
        if export_key is not None:
            data["exportKey"] = export_key
        if error is not None:
            data["error"] = error
        await self.publish_event(run_id, "export", data)

    async def publish_evaluation_event(
        self,
        run_id: str,
        status: Literal["success", "failed"],
        split: Literal["train", "validation", "test", "full"] | None = None,
        report_key: str | None = None,
        predictions_key: str | None = None,
        headline_metric: float | None = None,
        error: str | None = None,
    ) -> None:
        """Publish an evaluation-kind run event (RunEventSchema's fifth
        member — see server/lib/schema.ts). Best-effort: a failed/skipped
        evaluation still lets the run report `succeeded` — see
        `tasks/train.py`'s call site."""
        data: dict[str, Any] = {"status": status}
        if split is not None:
            data["split"] = split
        if report_key is not None:
            data["reportKey"] = report_key
        if predictions_key is not None:
            data["predictionsKey"] = predictions_key
        if headline_metric is not None:
            data["headlineMetric"] = headline_metric
        if error is not None:
            data["error"] = error
        await self.publish_event(run_id, "evaluation", data)

    # ──────────────────────────────────────────────────────────────
    # Abort flags (durable "last value wins" store, no separate KV pkg —
    # mirrors services/nats.ts's `setAbortFlag`/DLQ helpers)
    # ──────────────────────────────────────────────────────────────

    async def set_abort_flag(self, run_id: str) -> None:
        """Record that a run should be aborted. THESEUS_ABORT_FLAGS has
        `max_msgs_per_subject=1`, so this overwrites any previous flag."""
        await self.publish(subj.abort_flag(run_id), {"runId": run_id, "abortedAt": self._now()})

    async def is_aborted(self, run_id: str) -> bool:
        """Whether an abort flag has been recorded for this run. Backed by
        JetStream (not process memory), so it survives a worker restart —
        unlike the old in-memory `_abort_requests` set in tasks/train.py."""
        jsm = self.nc.jsm()
        try:
            await jsm.get_last_msg("THESEUS_ABORT_FLAGS", subj.abort_flag(run_id))
            return True
        except nats.js.errors.NotFoundError:
            return False

    # ──────────────────────────────────────────────────────────────
    # Dead-letter queue
    # ──────────────────────────────────────────────────────────────

    async def publish_to_dlq(
        self, kind: str, id: str, original_subject: str, payload: Any, error: str, delivery_count: int
    ) -> None:
        """Publish a permanently-failed message to THESEUS_DLQ for operator
        inspection/replay, instead of it silently expiring off THESEUS_TASKS
        after `max_deliver` retries with no trace anywhere."""
        await self.publish(
            subj.dlq(kind, id),
            {
                "originalSubject": original_subject,
                "payload": payload,
                "error": error,
                "deliveryCount": delivery_count,
            },
        )

    # ──────────────────────────────────────────────────────────────
    # Subscribing (pull-based consumers)
    # ──────────────────────────────────────────────────────────────

    # For Request - Reply
    async def subscribe(
        self,
        subject: str,
        handler: Callable[[Msg], Awaitable[None]],
        queue: str = "",
    ) -> None:
        """Basic Request-Reply subscription. Pass `queue` (a shared queue
        group name) so that if this worker is ever scaled to multiple
        replicas, only one replica handles each request instead of every
        replica receiving every message (core NATS `subscribe()` without a
        queue group fans out, it doesn't load-balance)."""
        await self.nc.subscribe(subject, queue=queue, cb=handler)
        logger.info(f"Subscribed to '{subject}' with request-reply (queue='{queue}')")

    async def _consume_loop[T: BaseModel](
        self,
        consumer: JetStreamContext.PullSubscription,
        subject: str,
        handler: Callable[[T], Awaitable[None]],
        message_type: type[T],
        retry_on_failure: bool = True,
        nak_delay: int = 10,
        fetch_timeout: int = 5,
        max_deliver: int = 1,
        dlq_kind: str | None = None,
        on_permanent_failure: Callable[[T, str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Generic pull consumer loop.

        A handler exception naks for redelivery up to `max_deliver` times
        (mirrors the consumer's own `max_deliver` — kept in sync by callers
        below). On the last attempt — or immediately for a message that can
        never succeed no matter how many retries (malformed JSON, or a
        well-formed payload that doesn't match `message_type`) — the message
        is acked instead of nak'd, `on_permanent_failure` runs (if given) so
        the caller can publish a terminal event exactly once, and a record
        goes to THESEUS_DLQ for operator visibility. Previously a task
        handler published its own "failed" event on every retry attempt
        (redundant) and a delivery-exhausted message just vanished silently
        once it aged off the stream (no DLQ at all).
        """
        while True:
            try:
                messages = await consumer.fetch(batch=1, timeout=fetch_timeout)
                for msg in messages:
                    raw_data: T | None = None
                    start = time.perf_counter()
                    try:
                        parent_ctx = propagate.extract(msg.headers or {})
                        raw_data = message_type.model_validate_json(msg.data.decode())
                        with tracer.start_as_current_span(
                            f"nats.consume {subject}", context=parent_ctx
                        ):
                            async with _ack_keepalive(msg):
                                await handler(raw_data)
                        await msg.ack()
                        nats_task_duration.record((time.perf_counter() - start) * 1000, {"subject": subject})
                        nats_task_count.add(1, {"subject": subject, "outcome": "success"})
                    except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as e:
                        # Can never succeed no matter how many retries.
                        logger.error(f"Malformed message on {subject}: {e}. Sending to DLQ.")
                        await msg.ack()
                        nats_task_count.add(1, {"subject": subject, "outcome": "malformed"})
                        if dlq_kind is not None:
                            await self.publish_to_dlq(
                                dlq_kind, msg.subject, msg.subject, msg.data.decode(errors="replace"), str(e), 1
                            )
                    except Exception as e:
                        num_delivered = msg.metadata.num_delivered
                        logger.exception(
                            f"Handler failed for {subject} (attempt {num_delivered}/{max_deliver})"
                        )
                        exhausted = not retry_on_failure or num_delivered >= max_deliver
                        nats_task_duration.record((time.perf_counter() - start) * 1000, {"subject": subject})
                        nats_task_count.add(
                            1, {"subject": subject, "outcome": "exhausted" if exhausted else "retry"}
                        )
                        if exhausted:
                            await msg.ack()
                            if raw_data is not None and on_permanent_failure is not None:
                                try:
                                    await on_permanent_failure(raw_data, str(e))
                                except Exception:
                                    logger.exception(f"on_permanent_failure callback failed for {subject}")
                            if dlq_kind is not None:
                                await self.publish_to_dlq(
                                    dlq_kind,
                                    msg.subject,
                                    msg.subject,
                                    raw_data.model_dump() if raw_data is not None else None,
                                    str(e),
                                    num_delivered,
                                )
                        else:
                            await msg.nak(delay=nak_delay)

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

    async def _reconcile_consumer(
        self, subject: str, durable_name: str, config: ConsumerConfig
    ) -> None:
        """Bring an already-existing durable consumer's config up to date.

        `pull_subscribe` creates a durable if it is absent but does NOT
        reconcile one that already exists — so changing `ack_wait` or
        `max_deliver` in code was a silent no-op against any environment that
        had run before. NATS treats CONSUMER.DURABLE.CREATE on an existing
        durable as an update for the fields that are updatable (both of these
        are), so re-issuing add_consumer is the reconciliation.
        """
        try:
            stream = await self.js.find_stream_name_by_subject(subject)
        except nats.js.errors.NotFoundError:
            logger.warning(f"No stream carries '{subject}'; skipping consumer reconciliation.")
            return

        jsm = self.nc.jsm()
        try:
            existing = await jsm.consumer_info(stream, durable_name)
        except nats.js.errors.NotFoundError:
            return  # pull_subscribe will create it

        current = existing.config
        if (
            current.ack_wait == config.ack_wait
            and current.max_deliver == config.max_deliver
            and current.filter_subject == config.filter_subject
        ):
            return

        logger.info(
            f"Reconciling consumer '{durable_name}': "
            f"ack_wait {current.ack_wait}->{config.ack_wait}, "
            f"max_deliver {current.max_deliver}->{config.max_deliver}, "
            f"filter_subject {current.filter_subject!r}->{config.filter_subject!r}"
        )
        try:
            await jsm.add_consumer(stream, config)
        except Exception:
            logger.exception(
                f"Could not update consumer '{durable_name}'. It will keep running with its old "
                f"config — delete the durable to pick up the new settings."
            )

    async def subscribe_tasks[T: BaseModel](
        self,
        subject: str,
        durable_name: str,
        handler: Callable[[T], Awaitable[None]],
        message_type: type[T],
        on_permanent_failure: Callable[[T, str], Awaitable[None]] | None = None,
        max_deliver: int = 3,
        nak_delay: int = 10,
    ) -> None:
        """
        Subscribe to task messages using a pull-based consumer.
        The handler receives the parsed JSON payload and should process it.
        Messages are acked after successful processing; nak'd on failure, up
        to `max_deliver` attempts, after which the message is given up on
        (see `_consume_loop`).
        """
        config = ConsumerConfig(
            durable_name=durable_name,
            filter_subject=subject,
            deliver_policy=DeliverPolicy.ALL,
            ack_wait=TASK_ACK_WAIT_SECONDS,
            max_deliver=max_deliver,
        )
        await self._reconcile_consumer(subject, durable_name, config)
        consumer = await self.js.pull_subscribe(subject, durable=durable_name, config=config)

        logger.info(
            f"Subscribed to '{subject}' as consumer '{durable_name}' "
            f"(ack_wait={TASK_ACK_WAIT_SECONDS}s, max_deliver={max_deliver})"
        )

        # Delegate to the generic loop (retries enabled)
        await self._consume_loop(
            consumer=consumer,
            subject=subject,
            handler=handler,
            message_type=message_type,
            retry_on_failure=True,
            nak_delay=nak_delay,
            max_deliver=max_deliver,
            dlq_kind=durable_name,
            on_permanent_failure=on_permanent_failure,
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
                filter_subject=subject,
                # THESEUS_COMMANDS is a workqueue-retention stream (server/lib/subjects.ts) —
                # NATS rejects any deliver_policy other than ALL there ("consumer must be
                # deliver all on workqueue stream", err_code=10101). DeliverPolicy.NEW was
                # never valid, so this consumer failed to create on every single startup;
                # the failure was silently swallowed because `run_task_workers()` runs as a
                # fire-and-forget `asyncio.create_task` (see main.py) whose exception is
                # never awaited/logged. A stale abort command from before the worker
                # connected is harmless to replay — `handle_command` only logs it, and the
                # actual guard is `is_aborted()` reading THESEUS_ABORT_FLAGS separately.
                deliver_policy=DeliverPolicy.ALL,
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
