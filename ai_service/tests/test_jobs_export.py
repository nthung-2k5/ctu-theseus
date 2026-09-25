"""Export jobs: conversion, assembly, retries and terminal states."""

from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from theseus.db.models import ModelExport
from theseus.jobs import export as export_job
from theseus.services import storage

# -- Export ----------------------------------------------------------------------------------


@pytest.fixture
def export_env(monkeypatch):
    log = SimpleNamespace(converted=[], bundled=[], artifact_exists=False)

    async def build_bundle(export_id):
        log.bundled.append(export_id)

    def convert(run_id, backend, artifact_id, dataset_key, export_id):
        log.converted.append((run_id, artifact_id, dataset_key))

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
