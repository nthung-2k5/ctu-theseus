"""Validate and enqueue inference jobs, and read their state back.

Shared by the session-cookie routes (/api/inference) and the API-key routes (/api/v1/predict) so
the two surfaces cannot drift on payload validation. Ported from server/lib/inference.ts.

Storage rule: durable storage only when the job outlives the request. A synchronous request
keeps its upload in a temp file; an asynchronous or batch one goes to the theseus-uploads bucket
so a restarted job can still find its input (see jobs/inference.py).
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from theseus import constants as C
from theseus.db.models import InferenceJob, TrainingRun
from theseus.jobs import inference as inference_jobs
from theseus.jobs.dispatcher import nudge
from theseus.services import storage
from theseus.services.task_registry import get_inference_input_spec
from theseus.services.training import new_uuid7
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass
class Dispatched:
    inference_id: uuid.UUID
    # Only set for a synchronous request: resolves when the job reaches a terminal state.
    waiter: asyncio.Future | None = None


@dataclass
class DispatchError:
    code: int
    message: str


def _parse_fields_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _read_upload(file: UploadFile) -> bytes | DispatchError:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return DispatchError(413, f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    return data


async def _store_input(inference_id: uuid.UUID, data: bytes, ext: str, *, durable: bool) -> tuple[dict, str | None]:
    """Persist an upload. Returns (payload fragment, upload_key)."""
    filename = f"{inference_id}{ext}"
    if durable:
        key = storage.inference_upload_key(str(inference_id), ext)
        await asyncio.get_running_loop().run_in_executor(
            None, storage.upload_bytes, C.BUCKET_UPLOADS, key, data, "application/octet-stream"
        )
        return {"filename": filename}, key
    folder = Path(get_settings().temp_dir) / "inference-uploads"
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / filename
    local.write_bytes(data)
    return {"filename": filename, "localPath": str(local)}, None


async def _insert_job(
    session: AsyncSession, job: InferenceJob, *, cleanup_key: str | None, cleanup_path: str | None
) -> None:
    try:
        session.add(job)
        await session.commit()
    except Exception:
        # The upload exists but no row points at it, so nothing would ever reap it.
        await session.rollback()
        if cleanup_key:
            await asyncio.get_running_loop().run_in_executor(None, storage.delete_file, C.BUCKET_UPLOADS, cleanup_key)
        if cleanup_path:
            Path(cleanup_path).unlink(missing_ok=True)
        raise
    nudge("inference")


async def dispatch_inference(
    session: AsyncSession,
    run: TrainingRun,
    task: str,
    *,
    file: UploadFile | None,
    fields: str | None,
    top_k: int | None,
    sync: bool = False,
) -> Dispatched | DispatchError:
    """Validate a single-item request against the run task and enqueue it."""
    if run.status != "succeeded":
        return DispatchError(409, "No successfully trained model found for this run")

    spec = get_inference_input_spec(task)
    inference_id = new_uuid7()
    payload: dict[str, Any]
    upload_key: str | None = None

    if spec["kind"] == "file":
        if file is None:
            return DispatchError(422, "This task requires a `file` field")
        accept = spec.get("accept")
        if accept and file.content_type not in accept:
            return DispatchError(422, f"This task only accepts: {', '.join(accept)}")
        data = await _read_upload(file)
        if isinstance(data, DispatchError):
            return data
        fragment, upload_key = await _store_input(
            inference_id, data, os.path.splitext(file.filename or "")[1], durable=not sync
        )
        payload = {"kind": "file", **fragment}
    else:
        if not fields:
            return DispatchError(422, "This task requires a `fields` field")
        parsed = _parse_fields_object(fields)
        if parsed is None:
            return DispatchError(422, "`fields` must be a JSON-encoded object")
        if spec["kind"] == "text":
            missing = [f for f in spec["fields"] if not isinstance(parsed.get(f), str) or parsed[f] == ""]
            if missing:
                return DispatchError(422, f"Missing required field(s): {', '.join(missing)}")
            payload = {"kind": "text", "fields": {f: parsed[f] for f in spec["fields"]}}
        else:
            for key, value in parsed.items():
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    return DispatchError(422, f"Field '{key}' must be a string or number")
            payload = {"kind": "record", "record": parsed}

    # Register the waiter BEFORE the job can possibly run.
    waiter = inference_jobs.register_waiter(inference_id) if sync else None
    job = InferenceJob(
        id=inference_id, run_id=run.id, status="pending", payload=payload, top_k=top_k, upload_key=upload_key
    )
    try:
        await _insert_job(session, job, cleanup_key=upload_key, cleanup_path=payload.get("localPath"))
    except Exception:
        inference_jobs.drop_waiter(inference_id)
        raise
    return Dispatched(inference_id, waiter)


async def dispatch_batch_inference(
    session: AsyncSession, run: TrainingRun, task: str, file: UploadFile
) -> Dispatched | DispatchError:
    """Validate and enqueue a batch (CSV of rows) job. Text and tabular tasks only."""
    if run.status != "succeeded":
        return DispatchError(409, "No successfully trained model found for this run")
    # File-backed tasks (vision/audio) would need an archive of many files, not a CSV of rows.
    if get_inference_input_spec(task)["kind"] == "file":
        return DispatchError(422, "Batch inference is only available for text and tabular tasks")
    data = await _read_upload(file)
    if isinstance(data, DispatchError):
        return data

    inference_id = new_uuid7()
    fragment, key = await _store_input(inference_id, data, ".csv", durable=True)
    job = InferenceJob(
        id=inference_id, run_id=run.id, status="pending", payload={"kind": "batch", **fragment}, upload_key=key
    )
    await _insert_job(session, job, cleanup_key=key, cleanup_path=None)
    return Dispatched(inference_id)


# -- Reading state back ----------------------------------------------------------------------


def public_status(status: str) -> str:
    """`running` is an internal state; clients only ever knew pending / success / failed."""
    return "pending" if status == "running" else status


def polled(job: InferenceJob) -> dict[str, Any]:
    """The polled shape: pending | success (+output) | batch (+rowCount) | failed (+error)."""
    if job.status in ("pending", "running"):
        return {"status": "pending"}
    if job.status == "failed":
        return {"status": "failed", "error": job.error or "Inference failed"}
    output = job.output or {}
    if output.get("kind") == "batch":
        return {"status": "batch", "rowCount": output["rowCount"]}
    return {"status": "success", "output": output}


async def get_job(
    session: AsyncSession, run_id: uuid.UUID, inference_id: uuid.UUID, *, refresh: bool = False
) -> InferenceJob | None:
    """Scoped by run in the same query, so an id belonging to another run simply does not match.

    refresh=True re-reads the row even if this session already holds it (a synchronous request
    inserted the job itself, and the worker has since updated it in another session).
    """
    stmt = sa.select(InferenceJob).where(InferenceJob.id == inference_id, InferenceJob.run_id == run_id)
    if refresh:
        stmt = stmt.execution_options(populate_existing=True)
    return (await session.execute(stmt)).scalar_one_or_none()


def batch_result_key(job: InferenceJob | None) -> str | None:
    if job is None or job.status != "success" or not job.output or job.output.get("kind") != "batch":
        return None
    return job.output["resultKey"]
