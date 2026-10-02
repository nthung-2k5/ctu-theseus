"""Run predictions inside the request that asked for them.

There is no job row and no queue: a prediction is validated, run on a worker thread while the request
waits, and its result is the response. Nothing about a prediction is stored. The trade-off is that a
server restart mid-request simply drops that request (the client retries), and there are no automatic
retries.

Shared by the session-cookie routes (/api/inference) and the API-key routes (/api/v1/predict), so the
two surfaces cannot drift on payload validation.

Concurrency: at most `inference_concurrency` predictions run at once. A request that cannot get a slot
within BUSY_WAIT_SECONDS is answered 503 + Retry-After rather than holding its connection open behind
a cold model load. A thread cannot be interrupted, so a slot is only freed once its thread has really
finished, even when the request already gave up on it.
"""

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile

from theseus.db.models import TrainingRun
from theseus.jobs.executors import run_in_executor
from theseus.services.predict import InferenceOutput, build_batch_result_frame
from theseus.services.task_registry import get_inference_input_spec
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Keeps one batch's memory and predict time bounded, and its result CSV a sane download size.
MAX_BATCH_ROWS = 10_000
BUSY_WAIT_SECONDS = 1.0
BUSY_RETRY_AFTER_SECONDS = 5

# The client only ever sees this; the real error (internal paths, keys) stays in the logs.
GENERIC_FAILURE = "Inference failed. Check server logs."


class InferenceInputError(ValueError):
    """The request is well-formed but its content cannot be scored. The message is safe to show the caller."""


@dataclass
class BatchResult:
    csv: bytes
    row_count: int


def _model_cache():
    """Imported lazily: the model cache pulls in torch and Ludwig, and the API must import without them."""
    from theseus.services.model_cache import get_model_cache

    return get_model_cache()


_background: set[asyncio.Task] = set()


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


# -- Concurrency -----------------------------------------------------------------------------

# One semaphore per event loop (a semaphore binds to the loop that first contends for it).
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(max(1, get_settings().inference_concurrency))
    return sem


async def _run_limited[T](work: Callable[[], Awaitable[T]]) -> T:
    """Run `work` in a concurrency slot, bounded by the inference timeout, mapping failures to HTTP errors."""
    sem = _semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), BUSY_WAIT_SECONDS)
    except TimeoutError:
        raise HTTPException(
            503,
            "The inference workers are busy, retry shortly",
            headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
        ) from None

    def _done(task: asyncio.Task) -> None:
        sem.release()
        if not task.cancelled():
            task.exception()  # mark retrieved: an abandoned task must not log "never retrieved"

    task = asyncio.ensure_future(work())
    task.add_done_callback(_done)
    try:
        # shield: a timeout or a client disconnect must not cancel work whose thread cannot be stopped.
        return await asyncio.wait_for(asyncio.shield(task), get_settings().inference_timeout_seconds)
    except TimeoutError:
        raise HTTPException(504, "Inference timed out") from None
    except InferenceInputError as e:
        raise HTTPException(422, str(e)) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(500, GENERIC_FAILURE) from None


# -- Input handling --------------------------------------------------------------------------


def _parse_fields_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    return data


def _safe_extension(filename: str | None) -> str:
    ext = os.path.splitext(filename or "")[1]
    return ext if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", ext) else ""


def _ensure_ready(run: TrainingRun) -> None:
    if run.status != "succeeded":
        raise HTTPException(409, "No successfully trained model found for this run")


async def _validated_input(task: str, *, file: UploadFile | None, fields: str | None) -> tuple[str, Any, str]:
    """Check a single-item request against the task. Returns (kind, value, file extension).

    `value` is the file bytes (kind "file"), the field values ("text") or the record ("record").
    """
    spec = get_inference_input_spec(task)
    if spec["kind"] == "file":
        if file is None:
            raise HTTPException(422, "This task requires a `file` field")
        accept = spec.get("accept")
        if accept and file.content_type not in accept:
            raise HTTPException(422, f"This task only accepts: {', '.join(accept)}")
        return "file", await _read_upload(file), _safe_extension(file.filename)

    if not fields:
        raise HTTPException(422, "This task requires a `fields` field")
    parsed = _parse_fields_object(fields)
    if parsed is None:
        raise HTTPException(422, "`fields` must be a JSON-encoded object")
    if spec["kind"] == "text":
        missing = [f for f in spec["fields"] if not isinstance(parsed.get(f), str) or parsed[f] == ""]
        if missing:
            raise HTTPException(422, f"Missing required field(s): {', '.join(missing)}")
        return "text", {f: parsed[f] for f in spec["fields"]}, ""
    for key, value in parsed.items():
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise HTTPException(422, f"Field '{key}' must be a string or number")
    return "record", parsed, ""


# -- Execution -------------------------------------------------------------------------------


async def predict_one(
    run: TrainingRun, task: str, *, file: UploadFile | None, fields: str | None, top_k: int | None
) -> InferenceOutput:
    """Validate and score one item (an image/audio file, a text object or a tabular record)."""
    _ensure_ready(run)
    kind, value, ext = await _validated_input(task, file=file, fields=fields)

    async def work() -> InferenceOutput:
        model = await _model_cache().get(str(run.id))
        input_columns = model.input_columns
        # Only meaningful for a single-input sequence task (token_classification): the tokens the
        # predicted tags align against.
        input_tokens: list[str] | None = None

        with tempfile.TemporaryDirectory() as temp_dir:
            if kind == "file":
                path = Path(temp_dir) / f"input{ext}"
                path.write_bytes(value)
                resolved: dict[str, Any] = {input_columns[0]: str(path)}
            elif kind == "text":
                missing = [c for c in input_columns if c not in value]
                if missing:
                    raise InferenceInputError(f"Missing required field(s): {', '.join(missing)}")
                resolved = {c: value[c] for c in input_columns}
                if len(input_columns) == 1:
                    input_tokens = str(resolved[input_columns[0]]).split()
            else:  # "record": tabular tasks have dataset-defined input features, not fixed ones
                resolved = dict(value)

            def _predict() -> InferenceOutput:
                frame = pd.DataFrame({col: [val] for col, val in resolved.items()})
                predictions = model.predict(frame)
                return model.to_output(predictions, top_k=top_k or 100, input_tokens=input_tokens)

            # The temp dir must outlive the thread, so the executor call stays inside this `with`.
            return await run_in_executor(None, _predict)

    return await _run_limited(work)


async def predict_batch(run: TrainingRun, task: str, file: UploadFile) -> BatchResult:
    """Score every row of an uploaded CSV in one predict call. Text and tabular tasks only."""
    _ensure_ready(run)
    # File-backed tasks (vision/audio) would need an archive of many files, not a CSV of rows.
    if get_inference_input_spec(task)["kind"] == "file":
        raise HTTPException(422, "Batch inference is only available for text and tabular tasks")
    data = await _read_upload(file)

    async def work() -> BatchResult:
        model = await _model_cache().get(str(run.id))

        def _predict_all() -> BatchResult:
            try:
                frame = pd.read_csv(io.BytesIO(data))
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
                raise InferenceInputError("The uploaded file is not a readable CSV") from None
            if len(frame) == 0:
                raise InferenceInputError("Uploaded batch file has no rows")
            if len(frame) > MAX_BATCH_ROWS:
                raise InferenceInputError(f"Batch file has {len(frame)} rows, exceeding the {MAX_BATCH_ROWS}-row limit")
            predictions = model.predict(frame)
            result = build_batch_result_frame(frame, predictions)
            return BatchResult(result.to_csv(index=False).encode(), len(result))

        return await run_in_executor(None, _predict_all)

    return await _run_limited(work)
