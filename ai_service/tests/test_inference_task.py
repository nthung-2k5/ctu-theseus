from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from schema.inference_task import InferenceTask, InferenceTaskPayload, InferenceTaskPayload1, InferenceTaskPayload3
from services.nats import nats_service
from tasks.inference import handle_inference_task, on_inference_permanent_failure


def _file_task() -> InferenceTask:
    return InferenceTask(
        inference_id=uuid4(),
        run_id=uuid4(),
        top_k=100,
        payload=InferenceTaskPayload(kind="file", upload_key="up-1", upload_filename="cat.png"),
    )


def _text_task() -> InferenceTask:
    return InferenceTask(
        inference_id=uuid4(),
        run_id=uuid4(),
        top_k=100,
        payload=InferenceTaskPayload1(kind="text", fields={"text": "hello"}),
    )


def _batch_task() -> InferenceTask:
    return InferenceTask(
        inference_id=uuid4(),
        run_id=uuid4(),
        top_k=100,
        payload=InferenceTaskPayload3(kind="batch", upload_key="up-batch", upload_filename="rows.csv"),
    )


@pytest.fixture
def patched_nats(monkeypatch):
    """`nats_service` is a module-level singleton imported by reference
    everywhere, so patching its methods here affects `tasks.inference` too
    — no real NATS connection needed for these tests."""
    publish = AsyncMock()
    delete_upload = AsyncMock()
    monkeypatch.setattr(nats_service, "publish", publish)
    monkeypatch.setattr(nats_service, "delete_upload", delete_upload)
    return publish, delete_upload


# ──────────────────────────────────────────────────────────────────
# on_inference_permanent_failure — fires once retries are exhausted
# ──────────────────────────────────────────────────────────────────


async def test_permanent_failure_publishes_a_generic_failed_result(patched_nats):
    publish, _ = patched_nats
    task = _text_task()

    await on_inference_permanent_failure(task, "actual internal error with a stack trace and an S3 key")

    publish.assert_awaited_once_with(
        f"theseus.inference.result.{task.inference_id}",
        {
            "status": "failed",
            "runId": str(task.run_id),
            "error": "Inference failed after multiple attempts — check server logs",
        },
    )


async def test_permanent_failure_deletes_the_upload_for_a_file_task(patched_nats):
    _, delete_upload = patched_nats
    task = _file_task()

    await on_inference_permanent_failure(task, "boom")

    delete_upload.assert_awaited_once_with("theseus-inferences", "up-1")


async def test_permanent_failure_does_not_touch_the_object_store_for_a_text_task(patched_nats):
    _, delete_upload = patched_nats
    task = _text_task()

    await on_inference_permanent_failure(task, "boom")

    delete_upload.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────
# handle_inference_task — the success path
# ──────────────────────────────────────────────────────────────────


async def test_successful_task_publishes_output_and_deletes_the_upload(monkeypatch, patched_nats):
    publish, delete_upload = patched_nats
    task = _file_task()
    fake_output = {"kind": "regression", "feature": "target", "value": 1.0}

    async def fake_run_inference_once(t, temp_dir):
        assert t is task
        return fake_output

    monkeypatch.setattr("tasks.inference._run_inference_once", fake_run_inference_once)

    await handle_inference_task(task)

    publish.assert_awaited_once_with(
        f"theseus.inference.result.{task.inference_id}",
        {"status": "success", "runId": str(task.run_id), "output": fake_output},
    )
    delete_upload.assert_awaited_once_with("theseus-inferences", "up-1")


async def test_successful_task_does_not_touch_object_store_for_non_file_payload(monkeypatch, patched_nats):
    publish, delete_upload = patched_nats
    task = _text_task()

    async def fake_run_inference_once(t, temp_dir):
        return {"kind": "text", "feature": "answer", "text": "hi"}

    monkeypatch.setattr("tasks.inference._run_inference_once", fake_run_inference_once)

    await handle_inference_task(task)

    publish.assert_awaited_once()
    delete_upload.assert_not_awaited()


async def test_a_failed_attempt_does_not_delete_the_upload_or_publish_a_result(monkeypatch, patched_nats):
    """A transient failure mid-attempt must leave the upload in place for a
    retry (`max_deliver=3`) and must not publish a terminal result — only
    `on_inference_permanent_failure` (after retries are exhausted) does
    either of those."""
    publish, delete_upload = patched_nats
    task = _file_task()

    async def failing_run_inference_once(t, temp_dir):
        raise RuntimeError("transient GPU error")

    monkeypatch.setattr("tasks.inference._run_inference_once", failing_run_inference_once)

    with pytest.raises(RuntimeError):
        await handle_inference_task(task)

    publish.assert_not_awaited()
    delete_upload.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────
# handle_inference_task — the batch branch (distinct result shape + upload)
# ──────────────────────────────────────────────────────────────────


async def test_batch_task_publishes_a_batch_result_and_deletes_the_upload(monkeypatch, patched_nats):
    publish, delete_upload = patched_nats
    task = _batch_task()
    upload_calls = []

    async def fake_run_batch_inference(t, temp_dir):
        assert t is task
        return "/tmp/fake_result.csv", 3

    monkeypatch.setattr("tasks.inference._run_batch_inference", fake_run_batch_inference)
    monkeypatch.setattr("tasks.inference.upload_file", lambda bucket, key, path: upload_calls.append((bucket, key, path)))

    await handle_inference_task(task)

    expected_key = f"{task.run_id}/predictions/{task.inference_id}.csv"
    publish.assert_awaited_once_with(
        f"theseus.inference.result.{task.inference_id}",
        {"status": "batch", "runId": str(task.run_id), "resultKey": expected_key, "rowCount": 3},
    )
    assert upload_calls == [("theseus-models", expected_key, "/tmp/fake_result.csv")]
    delete_upload.assert_awaited_once_with("theseus-inferences", "up-batch")


async def test_batch_task_failure_does_not_publish_or_delete_the_upload(monkeypatch, patched_nats):
    publish, delete_upload = patched_nats
    task = _batch_task()

    async def failing_run_batch_inference(t, temp_dir):
        raise ValueError("Uploaded batch file has no rows")

    monkeypatch.setattr("tasks.inference._run_batch_inference", failing_run_batch_inference)

    with pytest.raises(ValueError):
        await handle_inference_task(task)

    publish.assert_not_awaited()
    delete_upload.assert_not_awaited()
