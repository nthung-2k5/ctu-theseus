import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from config import INFERENCE_TIMEOUT_SECONDS
from constants import BUCKET_MODELS
from nats.aio.msg import Msg
from opentelemetry import trace
from schema.inference_task import InferenceTask
from services.model_cache import model_cache
from services.nats import nats_service
from services.predict import InferenceOutput, build_batch_result_frame, build_inference_output
from services.storage import batch_inference_result_key, upload_file

from schema import subjects as subj

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("theseus-worker")

# Keeps one batch job's memory/predict-time bounded, and keeps a batch result
# CSV well within what's reasonable to upload/download over HTTP.
MAX_BATCH_ROWS = 10_000


async def _resolve_file_input(upload_key: str, upload_filename: str, temp_dir: str) -> str:
    """Download a file-backed payload's upload from the NATS Object Store to
    a local path. Does *not* delete it — a task on THESEUS_TASKS may be
    redelivered (`max_deliver=3`) after a transient failure, and a retry
    needs the upload to still be there. The caller deletes it once the
    job's outcome is final (success, or permanently failed) — see
    `handle_inference_task` / `on_inference_permanent_failure`."""
    local_path = Path(temp_dir) / upload_filename
    downloaded = await nats_service.download("theseus-inferences", upload_key, local_path)
    if not downloaded:
        raise FileNotFoundError(f"Upload not found for key: {upload_key}")
    return str(local_path)


async def _run_inference_once(task: InferenceTask, temp_dir: str) -> InferenceOutput:
    model = await model_cache.get(str(task.run_id))
    input_features = model.config_obj.input_features
    output_feature = model.config_obj.output_features[0]

    payload = task.payload
    # Only meaningful for a single-input sequence task (token_classification)
    # — the tokens a `tokens`-kind output aligns its predicted tags against.
    # A multi-input text task (question_answering) never has a sequence
    # output, so this stays unused there.
    input_tokens: list[str] | None = None

    if payload.kind == "file":
        local_path = await _resolve_file_input(payload.upload_key, payload.upload_filename, temp_dir)
        resolved_input: dict[str, Any] = {input_features[0].column: local_path}
    elif payload.kind == "text":
        missing = [f.column for f in input_features if f.column not in payload.fields]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")
        resolved_input = {f.column: payload.fields[f.column] for f in input_features}
        if len(input_features) == 1:
            input_tokens = str(resolved_input[input_features[0].column]).split()
    else:  # "record" — tabular tasks have dataset-defined input features, not fixed ones.
        resolved_input = dict(payload.record)

    def _predict():
        with tracer.start_as_current_span("inference.predict"):
            input_df = pd.DataFrame({col: [val] for col, val in resolved_input.items()})
            predictions, _ = model.predict(dataset=input_df)
            assert isinstance(predictions, pd.DataFrame)

            idx2str = None
            if output_feature.type == "category" and model.training_set_metadata:
                idx2str = model.training_set_metadata.get(output_feature.name, {}).get("idx2str")

            return output_feature.name, output_feature.type, predictions, idx2str

    feature_name, feature_type, predictions, idx2str = await asyncio.to_thread(_predict)

    return build_inference_output(
        feature_name,
        feature_type,
        predictions,
        idx2str,
        top_k=task.top_k or 100,
        input_tokens=input_tokens,
    )


async def _run_batch_inference(task: InferenceTask, temp_dir: str) -> tuple[str, int]:
    """Score every row of an uploaded CSV in one `model.predict` call —
    what Ludwig is actually efficient at, unlike `_run_inference_once`'s
    one-row-per-call path. Returns (local result CSV path, row count).
    """
    model = await model_cache.get(str(task.run_id))
    payload = task.payload
    assert payload.kind == "batch"

    local_upload_path = await _resolve_file_input(payload.upload_key, payload.upload_filename, temp_dir)

    def _predict_and_write() -> tuple[str, int]:
        with tracer.start_as_current_span("inference.predict_batch"):
            input_df = pd.read_csv(local_upload_path)
            if len(input_df) == 0:
                raise ValueError("Uploaded batch file has no rows")
            if len(input_df) > MAX_BATCH_ROWS:
                raise ValueError(f"Batch file has {len(input_df)} rows, exceeding the {MAX_BATCH_ROWS}-row limit")

            predictions, _ = model.predict(dataset=input_df)
            assert isinstance(predictions, pd.DataFrame)
            result_df = build_batch_result_frame(input_df, predictions)

            result_path = str(Path(temp_dir) / "batch_result.csv")
            result_df.to_csv(result_path, index=False)
            return result_path, len(result_df)

    return await asyncio.to_thread(_predict_and_write)


async def _delete_upload_if_file(task: InferenceTask) -> None:
    """Best-effort cleanup once a job's outcome is final — a task that will
    be retried must *not* have its upload deleted yet (see
    `_resolve_file_input`)."""
    if task.payload.kind in ("file", "batch"):
        await nats_service.delete_upload("theseus-inferences", task.payload.upload_key)


async def handle_inference_task(task: InferenceTask) -> None:
    """Handle one dispatched inference job (JetStream pull consumer, via
    `subscribe_tasks` — see tasks/__init__.py). Unlike the old core-NATS
    request/reply this replaced, a raised exception here is handled by
    `_consume_loop`'s retry/DLQ machinery, not by this function — it must
    publish the terminal result exactly once, on the attempt that actually
    finishes (success) or on `on_inference_permanent_failure` (retries
    exhausted), not on every failing attempt in between.
    """
    inference_id = str(task.inference_id)
    run_id = str(task.run_id)
    logger.info(f"Starting inference job {inference_id} for run {run_id}")

    with tempfile.TemporaryDirectory() as temp_dir:
        if task.payload.kind == "batch":
            result_path, row_count = await asyncio.wait_for(
                _run_batch_inference(task, temp_dir), timeout=INFERENCE_TIMEOUT_SECONDS
            )
            result_key = batch_inference_result_key(run_id, inference_id)
            await asyncio.to_thread(upload_file, BUCKET_MODELS, result_key, result_path)
            await nats_service.publish(
                subj.inference_result(inference_id),
                {"status": "batch", "runId": run_id, "resultKey": result_key, "rowCount": row_count},
            )
        else:
            output = await asyncio.wait_for(_run_inference_once(task, temp_dir), timeout=INFERENCE_TIMEOUT_SECONDS)
            await nats_service.publish(
                subj.inference_result(inference_id), {"status": "success", "runId": run_id, "output": output}
            )

    await _delete_upload_if_file(task)
    logger.info(f"Inference job {inference_id} completed")


async def on_inference_permanent_failure(task: InferenceTask, error: str) -> None:
    """Called once an `InferenceTask` has exhausted all delivery attempts
    (see `services/nats.py` `_consume_loop`'s `on_permanent_failure`) —
    publishes the terminal `failed` result exactly once, so a client
    polling `theseus.inference.result.{inferenceId}` doesn't wait out the
    stream's full 1-hour retention for nothing. The real error (including
    any internal paths/S3 keys) is already in the worker's logs and the
    THESEUS_DLQ record `_consume_loop` publishes alongside this call; the
    client only gets a generic message."""
    inference_id = str(task.inference_id)
    run_id = str(task.run_id)
    logger.error(f"Inference job {inference_id} failed permanently: {error}")
    await nats_service.publish(
        subj.inference_result(inference_id),
        {
            "status": "failed",
            "runId": run_id,
            "error": "Inference failed after multiple attempts — check server logs",
        },
    )
    await _delete_upload_if_file(task)


async def handle_inference_warm(msg: Msg) -> None:
    """Preload a run's model into the cache ahead of the user's first real
    request (see server/lib/nats.ts's `publishInferenceWarm`). Fire-and-
    forget core NATS publish with no payload — the run id is the subject's
    trailing token (`theseus.inference.warm.{runId}`), and there's no reply
    to send."""
    run_id = msg.subject.rsplit(".", 1)[-1]
    try:
        await model_cache.warm(run_id)
        logger.info(f"Warmed inference model cache for run {run_id}")
    except Exception:
        logger.exception(f"Failed to warm model cache for run {run_id}")
