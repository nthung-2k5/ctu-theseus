"""Inference and export jobs: input handling, retries, terminal states, cleanup."""

import asyncio
import time
from types import SimpleNamespace

import pandas as pd
import pytest
import sqlalchemy as sa

from theseus.db.models import InferenceJob, ModelExport
from theseus.jobs import export as export_job
from theseus.jobs import inference as inf
from theseus.services import storage
from theseus.settings import get_settings


class FakeModel:
    """Enough of LudwigModel for the job mechanics; the prediction shaping is tested in test_predict."""

    predict_calls: list[pd.DataFrame] = []
    delay = 0.0

    def __init__(self, columns=("text",), out=("class", "category")):
        self.config_obj = SimpleNamespace(
            input_features=[SimpleNamespace(column=c) for c in columns],
            output_features=[SimpleNamespace(name=out[0], type=out[1])],
        )
        self.training_set_metadata = {}

    def predict(self, dataset):
        FakeModel.predict_calls.append(dataset)
        time.sleep(FakeModel.delay)
        return pd.DataFrame({"class_predictions": ["cat"] * len(dataset)}), None


@pytest.fixture
def serving(monkeypatch, tmp_path):
    FakeModel.predict_calls, FakeModel.delay = [], 0.0
    model = FakeModel()
    log = SimpleNamespace(model=model, downloads=[], uploads={}, deleted=[], cache_requests=[])

    class Cache:
        async def get(self, run_id):
            log.cache_requests.append(run_id)
            return log.model

    monkeypatch.setattr(inf, "_model_cache", lambda: Cache())
    monkeypatch.setattr(
        inf, "build_inference_output", lambda *a, **k: {"kind": "classification", "feature": "class", "classes": []}
    )
    monkeypatch.setattr(inf, "build_batch_result_frame", lambda frame, preds: frame.assign(prediction="cat"))
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)

    def download(bucket, key, dest):
        log.downloads.append((bucket, key))
        with open(dest, "w") as f:
            f.write("a,b\n1,2\n3,4\n")
        return dest

    monkeypatch.setattr(storage, "download_file", download)
    monkeypatch.setattr(
        storage, "upload_file", lambda bucket, key, path: log.uploads.__setitem__((bucket, key), open(path).read())
    )
    monkeypatch.setattr(storage, "delete_file", lambda bucket, key: log.deleted.append((bucket, key)))
    return log


async def job(db, job_id):
    async with db() as s:
        return (await s.execute(sa.select(InferenceJob).where(InferenceJob.id == job_id))).scalar_one()


async def running(make_inference, **kw):
    kw.setdefault("attempt", 1)
    return await make_inference(status="running", **kw)


# -- Inference: success paths ----------------------------------------------------------------


async def test_a_text_job_succeeds_stores_its_output_and_resolves_the_waiter(db, serving, make_inference):
    job_id, run_id = await running(make_inference, payload={"kind": "text", "fields": {"text": "great movie"}})
    waiter = inf.register_waiter(job_id)

    await inf.run_inference(job_id)

    j = await job(db, job_id)
    assert j.status == "success" and j.completed_at is not None and j.claimed_by is None
    assert j.output == {"kind": "classification", "feature": "class", "classes": []}
    assert serving.cache_requests == [str(run_id)]
    assert FakeModel.predict_calls[0].to_dict("records") == [{"text": "great movie"}]
    assert await asyncio.wait_for(waiter, 1) == "success"


async def test_a_text_job_missing_a_required_field_fails_with_a_clear_message(db, serving, make_inference):
    serving.model = FakeModel(columns=("context", "question"))
    job_id, _ = await running(make_inference, payload={"kind": "text", "fields": {"context": "c"}})
    with pytest.raises(ValueError, match="Missing required field.*question"):
        await inf.run_inference(job_id)


async def test_a_record_job_passes_the_record_columns_straight_through(db, serving, make_inference):
    job_id, _ = await running(make_inference, payload={"kind": "record", "record": {"age": 30, "income": 5}})
    await inf.run_inference(job_id)
    assert FakeModel.predict_calls[0].to_dict("records") == [{"age": 30, "income": 5}]


async def test_a_file_job_downloads_its_upload_then_deletes_it_and_forgets_the_key(db, serving, make_inference):
    job_id, _ = await running(
        make_inference, payload={"kind": "file", "filename": "cat.png"}, upload_key="inference/abc/input.png"
    )
    serving.model = FakeModel(columns=("image_path",))
    await inf.run_inference(job_id)

    assert serving.downloads == [("theseus-uploads", "inference/abc/input.png")]
    assert FakeModel.predict_calls[0].columns.tolist() == ["image_path"]
    assert serving.deleted == [("theseus-uploads", "inference/abc/input.png")]  # deleted only once the outcome is final
    assert (await job(db, job_id)).upload_key is None


async def test_a_synchronous_file_job_uses_its_temp_file_with_no_s3_and_removes_it(
    db, serving, make_inference, tmp_path
):
    local = tmp_path / "upload.png"
    local.write_bytes(b"png")
    job_id, _ = await running(make_inference, payload={"kind": "file", "filename": "x.png", "localPath": str(local)})
    serving.model = FakeModel(columns=("image_path",))

    await inf.run_inference(job_id)

    assert serving.downloads == [] and serving.deleted == []  # never touched S3
    assert FakeModel.predict_calls[0]["image_path"][0] == str(local)
    assert not local.exists()


async def test_a_batch_job_scores_every_row_and_uploads_the_result_csv(db, serving, make_inference):
    job_id, run_id = await running(
        make_inference, payload={"kind": "batch", "filename": "rows.csv"}, upload_key="inference/b/input.csv"
    )
    await inf.run_inference(job_id)

    j = await job(db, job_id)
    result_key = f"{run_id}/predictions/{job_id}.csv"
    assert j.status == "success" and j.output == {"kind": "batch", "resultKey": result_key, "rowCount": 2}
    assert "prediction" in serving.uploads[("theseus-models", result_key)]
    assert len(FakeModel.predict_calls) == 1  # one predict call for the whole file, not one per row


async def test_a_batch_over_the_row_limit_is_rejected(db, serving, make_inference, monkeypatch):
    monkeypatch.setattr(inf, "MAX_BATCH_ROWS", 1)
    job_id, _ = await running(
        make_inference, payload={"kind": "batch", "filename": "r.csv"}, upload_key="inference/b/i.csv"
    )
    with pytest.raises(ValueError, match="exceeding the 1-row limit"):
        await inf.run_inference(job_id)


# -- Inference: failure, retry, races --------------------------------------------------------


async def test_a_failed_attempt_is_retried_and_keeps_its_upload_until_the_outcome_is_final(db, serving, make_inference):
    job_id, _ = await running(
        make_inference, payload={"kind": "batch", "filename": "r.csv"}, upload_key="inference/k/i.csv", attempt=1
    )
    await inf.handle_failure(job_id, RuntimeError("transient S3 hiccup"))

    j = await job(db, job_id)
    assert j.status == "pending" and j.last_error == "RuntimeError: transient S3 hiccup"
    assert j.upload_key == "inference/k/i.csv" and serving.deleted == []  # the retry still needs the input


async def test_the_last_failed_attempt_fails_the_job_hides_internals_and_cleans_up(db, serving, make_inference):
    job_id, _ = await running(
        make_inference, payload={"kind": "batch", "filename": "r.csv"}, upload_key="inference/k/i.csv", attempt=3
    )
    waiter = inf.register_waiter(job_id)

    await inf.handle_failure(job_id, RuntimeError("s3://theseus-uploads/inference/k/i.csv: AccessDenied"))

    j = await job(db, job_id)
    assert j.status == "failed" and j.completed_at is not None
    assert j.error == inf.GENERIC_FAILURE and "s3://" not in j.error  # the client never sees internal keys
    assert "AccessDenied" in j.last_error  # but operators do
    assert serving.deleted == [("theseus-uploads", "inference/k/i.csv")]
    assert await asyncio.wait_for(waiter, 1) == "failed"


async def test_a_job_that_exceeds_the_timeout_fails_through_the_normal_path(db, serving, make_inference, monkeypatch):
    monkeypatch.setattr(get_settings(), "inference_timeout_seconds", 0.1)
    FakeModel.delay = 0.6
    job_id, _ = await running(make_inference)
    with pytest.raises(TimeoutError):
        await inf.run_inference(job_id)


async def test_a_late_result_cannot_overwrite_a_job_the_reaper_already_failed(db, serving, make_inference):
    job_id, _ = await make_inference(status="failed", attempt=1, error="x")
    assert await inf._finish_success(job_id, {"kind": "text"}) is False
    j = await job(db, job_id)
    assert j.status == "failed" and j.output is None


async def test_waiters_are_registered_before_enqueue_and_can_be_dropped(db):
    job_id = __import__("uuid").uuid4()
    fut = inf.register_waiter(job_id)
    inf.drop_waiter(job_id)
    inf._resolve_waiter(job_id, "success")  # nobody is waiting any more: must not raise
    assert not fut.done()


# -- Export ----------------------------------------------------------------------------------


@pytest.fixture
def export_env(monkeypatch):
    log = SimpleNamespace(converted=[], bundled=[], artifact_exists=False)

    async def build_bundle(export_id):
        log.bundled.append(export_id)

    def convert(run_id, artifact, dataset_key, export_id):
        log.converted.append((run_id, artifact.id, dataset_key))

    monkeypatch.setattr(export_job.bundle, "build_bundle", build_bundle)
    monkeypatch.setattr(export_job, "_convert", convert)
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: log.artifact_exists)
    return log


async def export_row(db, export_id):
    async with db() as s:
        return (await s.execute(sa.select(ModelExport).where(ModelExport.id == export_id))).scalar_one()


async def test_export_converts_a_missing_artifact_then_assembles(db, export_env, make_export):
    export_id, run_id = await make_export(status="converting", attempt=1)
    await export_job.run_export(export_id)

    assert [(r, f) for r, f, _ in export_env.converted] == [(str(run_id), "onnx")]
    assert export_env.converted[0][2].startswith("snapshots/")  # golden sample reads the run's snapshot
    assert export_env.bundled == [export_id]
    assert (await export_row(db, export_id)).status == "assembling"  # build_bundle owns ready / failed


async def test_export_converts_the_artifact_its_format_is_built_from(db, export_env, make_export):
    export_id, run_id = await make_export(status="converting", attempt=1, fmt="torch_export")
    await export_job.run_export(export_id)
    assert [(r, a) for r, a, _ in export_env.converted] == [(str(run_id), "torch_export")]


async def test_export_skips_conversion_when_another_format_already_produced_the_artifact(db, export_env, make_export):
    export_env.artifact_exists = True
    export_id, _ = await make_export(status="converting", attempt=1, fmt="python_devkit")
    await export_job.run_export(export_id)
    assert export_env.converted == [] and export_env.bundled == [export_id]


async def test_an_export_whose_format_was_uninstalled_raises_so_the_dispatcher_fails_it(db, export_env, make_export):
    export_id, _ = await make_export(status="converting", attempt=1, fmt="removed_format")
    with pytest.raises(ValueError, match="'removed_format' is no longer installed"):
        await export_job.run_export(export_id)
    assert export_env.converted == [] and export_env.bundled == []


async def test_export_does_not_assemble_a_job_that_was_recovered_while_it_converted(db, export_env, make_export):
    export_id, _ = await make_export(status="pending", attempt=1)  # startup recovery re-queued it under our feet
    await export_job.run_export(export_id)
    assert export_env.bundled == []
    assert (await export_row(db, export_id)).status == "pending"


async def test_a_failed_export_attempt_is_retried_then_fails_for_good_with_its_message(db, export_env, make_export):
    export_id, _ = await make_export(status="converting", attempt=1)
    await export_job.handle_failure(export_id, RuntimeError("torch export failed"))
    assert (await export_row(db, export_id)).status == "pending"

    async with db() as s:
        await s.execute(
            sa.update(ModelExport).where(ModelExport.id == export_id).values(status="assembling", attempt=3)
        )
        await s.commit()
    await export_job.handle_failure(export_id, RuntimeError("torch export failed"))
    e = await export_row(db, export_id)
    assert e.status == "failed" and e.failed_message == "RuntimeError: torch export failed"
