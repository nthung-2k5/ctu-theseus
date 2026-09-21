"""Run one inference job (single row, or a batch CSV).

Replaces ai_service/tasks/inference.py. Input rule: durable storage only when the job outlives
the request. A synchronous predict keeps its upload in a temp file (payload.localPath); an
asynchronous or batch job stores it in the theseus-uploads bucket (row.upload_key) so a
restarted job can still find it. The NATS object store and its stale-blob reaper are gone.

The payload on the row is one of:
    {"kind": "file",   "filename": "x.png", "localPath": optional}
    {"kind": "text",   "fields": {...}}
    {"kind": "record", "record": {...}}
    {"kind": "batch",  "filename": "x.csv", "localPath": optional}
"""

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
from opentelemetry import trace

from theseus import constants as C
from theseus.db.base import get_sessionmaker
from theseus.db.models import InferenceJob
from theseus.jobs import queue
from theseus.jobs.executors import run_in_executor
from theseus.services import storage
from theseus.services.predict import InferenceOutput, build_batch_result_frame, build_inference_output
from theseus.settings import get_settings

logger = logging.getLogger("theseus.jobs.inference")
tracer = trace.get_tracer("theseus")


def _model_cache():
    """Imported lazily: the model cache pulls in torch and Ludwig, and the API must import without them."""
    from theseus.services.model_cache import get_model_cache

    return get_model_cache()


# Keeps one batch job memory and predict time bounded, and its result CSV a sane download size.
MAX_BATCH_ROWS = 10_000

# The client only ever sees this; the real error (internal paths, S3 keys) stays in last_error and the logs.
GENERIC_FAILURE = "Inference failed after multiple attempts. Check server logs."

_waiters: dict[uuid.UUID, asyncio.Future] = {}
_background: set[asyncio.Task] = set()


# -- Synchronous-wait support ----------------------------------------------------------------


def register_waiter(job_id: uuid.UUID) -> asyncio.Future:
    """Call BEFORE enqueueing, so a fast job cannot finish before anyone is waiting on it."""
    fut = asyncio.get_running_loop().create_future()
    _waiters[job_id] = fut
    return fut


def drop_waiter(job_id: uuid.UUID) -> None:
    _waiters.pop(job_id, None)


def _resolve_waiter(job_id: uuid.UUID, status: str) -> None:
    fut = _waiters.pop(job_id, None)
    if fut is not None and not fut.done():
        fut.set_result(status)


def spawn_warm(run_id: str) -> None:
    """Preload a run model into the cache ahead of the first real request (fire and forget)."""

    async def _warm() -> None:
        try:
            await _model_cache().warm(run_id)
            logger.info("Warmed inference model cache for run %s", run_id)
        except Exception:
            logger.exception("Failed to warm model cache for run %s", run_id)

    task = asyncio.create_task(_warm())
    _background.add(task)  # strong reference: the loop only keeps a weak one
    task.add_done_callback(_background.discard)


# -- Execution -------------------------------------------------------------------------------


async def _resolve_file_input(job: InferenceJob, temp_dir: str) -> str:
    """A local path to the job upload. Never deletes it: the caller does that once the outcome is final."""
    payload = job.payload
    local = payload.get("localPath")
    if local and os.path.exists(local):
        return local
    if job.upload_key:
        dest = str(Path(temp_dir) / (payload.get("filename") or "input"))
        return await run_in_executor(None, storage.download_file, C.BUCKET_UPLOADS, job.upload_key, dest)
    raise FileNotFoundError("The inference upload is no longer available")


async def _run_single(job: InferenceJob, temp_dir: str) -> InferenceOutput:
    model = await _model_cache().get(str(job.run_id))
    input_features = model.config_obj.input_features
    output_feature = model.config_obj.output_features[0]
    payload = job.payload
    # Only meaningful for a single-input sequence task (token_classification): the tokens the
    # predicted tags align against.
    input_tokens: list[str] | None = None

    if payload["kind"] == "file":
        resolved: dict[str, Any] = {input_features[0].column: await _resolve_file_input(job, temp_dir)}
    elif payload["kind"] == "text":
        fields = payload["fields"]
        missing = [f.column for f in input_features if f.column not in fields]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")
        resolved = {f.column: fields[f.column] for f in input_features}
        if len(input_features) == 1:
            input_tokens = str(resolved[input_features[0].column]).split()
    else:  # "record": tabular tasks have dataset-defined input features, not fixed ones
        resolved = dict(payload["record"])

    def _predict():
        with tracer.start_as_current_span("inference.predict"):
            frame = pd.DataFrame({col: [val] for col, val in resolved.items()})
            predictions, _ = model.predict(dataset=frame)
            assert isinstance(predictions, pd.DataFrame)
            idx2str = None
            if output_feature.type == "category" and model.training_set_metadata:
                idx2str = model.training_set_metadata.get(output_feature.name, {}).get("idx2str")
            return predictions, idx2str

    predictions, idx2str = await run_in_executor(None, _predict)
    return build_inference_output(
        output_feature.name,
        output_feature.type,
        predictions,
        idx2str,
        top_k=job.top_k or 100,
        input_tokens=input_tokens,
    )


async def _run_batch(job: InferenceJob, temp_dir: str) -> tuple[str, int]:
    """Score every row of an uploaded CSV in one model.predict call. Returns (result path, rows)."""
    model = await _model_cache().get(str(job.run_id))
    upload_path = await _resolve_file_input(job, temp_dir)

    def _predict_and_write() -> tuple[str, int]:
        with tracer.start_as_current_span("inference.predict_batch"):
            frame = pd.read_csv(upload_path)
            if len(frame) == 0:
                raise ValueError("Uploaded batch file has no rows")
            if len(frame) > MAX_BATCH_ROWS:
                raise ValueError(f"Batch file has {len(frame)} rows, exceeding the {MAX_BATCH_ROWS}-row limit")
            predictions, _ = model.predict(dataset=frame)
            assert isinstance(predictions, pd.DataFrame)
            result = build_batch_result_frame(frame, predictions)
            path = str(Path(temp_dir) / "batch_result.csv")
            result.to_csv(path, index=False)
            return path, len(result)

    return await run_in_executor(None, _predict_and_write)


async def _finish_success(job_id: uuid.UUID, output: Any) -> bool:
    async with get_sessionmaker()() as s:
        res = await s.execute(
            sa.update(InferenceJob)
            .where(InferenceJob.id == job_id, InferenceJob.status == "running")
            .values(status="success", output=output, completed_at=sa.func.now(), claimed_by=None, lease_expires_at=None)
            .returning(InferenceJob.id)
        )
        won = res.first() is not None
        await s.commit()
    return won


async def cleanup_upload(job_id: uuid.UUID) -> None:
    """Delete a finished job upload (S3 object and local temp file) and forget its key. Best effort."""
    async with get_sessionmaker()() as s:
        job = await s.get(InferenceJob, job_id)
        if job is None:
            return
        local = (job.payload or {}).get("localPath")
        key = job.upload_key
    if local:
        Path(local).unlink(missing_ok=True)
    if key:
        try:
            await run_in_executor(None, storage.delete_file, C.BUCKET_UPLOADS, key)
        except Exception:
            logger.warning("Could not delete inference upload %s", key, exc_info=True)
            return
        async with get_sessionmaker()() as s:
            await s.execute(sa.update(InferenceJob).where(InferenceJob.id == job_id).values(upload_key=None))
            await s.commit()


async def run_inference(job_id: uuid.UUID) -> None:
    """Run a claimed inference job. Raises on failure; the dispatcher then calls handle_failure."""
    async with get_sessionmaker()() as s:
        job = await s.get(InferenceJob, job_id)
    if job is None:
        return
    logger.info("Starting inference job %s for run %s", job_id, job.run_id)
    timeout = get_settings().inference_timeout_seconds

    with tempfile.TemporaryDirectory() as temp_dir:
        if job.payload["kind"] == "batch":
            result_path, row_count = await asyncio.wait_for(_run_batch(job, temp_dir), timeout)
            result_key = storage.batch_inference_result_key(str(job.run_id), str(job.id))
            await run_in_executor(None, storage.upload_file, C.BUCKET_MODELS, result_key, result_path)
            output: Any = {"kind": "batch", "resultKey": result_key, "rowCount": row_count}
        else:
            output = await asyncio.wait_for(_run_single(job, temp_dir), timeout)

    if await _finish_success(job_id, output):
        await cleanup_upload(job_id)
    _resolve_waiter(job_id, "success")
    logger.info("Inference job %s completed", job_id)


async def handle_failure(job_id: uuid.UUID, exc: BaseException) -> None:
    """An attempt failed: retry after a delay, or fail for good once attempts are exhausted."""
    error = f"{type(exc).__name__}: {exc}"
    logger.error("Inference job %s failed: %s", job_id, error)
    new_status = await queue.release_or_fail(
        queue.INFERENCE,
        job_id,
        error,
        final_values={"error": GENERIC_FAILURE, "completed_at": sa.func.now()},
    )
    if new_status == "failed":
        await cleanup_upload(job_id)
        _resolve_waiter(job_id, "failed")
