import uuid

import pytest

from theseus.db.models import Dataset, DatasetItem, DatasetVersion, InferenceJob, Project, TrainingRun, User
from theseus.services import cleanup, storage


@pytest.fixture
def calls(monkeypatch):
    """Record every storage delete instead of touching S3."""
    recorded: list[tuple] = []
    monkeypatch.setattr(storage, "delete_file", lambda b, k: recorded.append(("file", b, k)))
    monkeypatch.setattr(storage, "delete_files", lambda b, ks: recorded.append(("files", b, tuple(sorted(ks)))))
    monkeypatch.setattr(storage, "delete_prefix", lambda b, p: recorded.append(("prefix", b, p)))
    return recorded


async def test_version_cleanup_deletes_snapshot_objects_but_is_a_noop_for_the_draft(calls):
    vid = uuid.uuid4()
    await cleanup.cleanup_version_storage(vid, None)
    assert calls == []
    await cleanup.cleanup_version_storage(vid, "v1")
    assert sorted(calls) == sorted(
        [
            ("file", "theseus-datasets", f"snapshots/{vid}/dataset.parquet"),
            ("file", "theseus-datasets", f"snapshots/{vid}/manifest.json"),
            ("prefix", "theseus-datasets", f"snapshots/{vid}/augmented/"),  # copies built with the snapshot
        ]
    )


async def test_failed_delete_is_swallowed_and_does_not_stop_the_rest(monkeypatch):
    seen = []

    def flaky(bucket, key):
        seen.append(key)
        raise RuntimeError("boom")

    monkeypatch.setattr(storage, "delete_file", flaky)
    monkeypatch.setattr(storage, "delete_prefix", flaky)  # same (bucket, key-or-prefix) shape
    await cleanup.cleanup_version_storage(uuid.uuid4(), "v1")  # must not raise
    assert len(seen) == 3  # parquet, manifest and the augmented prefix were all still attempted


async def _seed_project(s):
    user = User(name="U", email=f"{uuid.uuid4()}@x.co", password_hash="x")
    s.add(user)
    await s.flush()
    project = Project(user_id=user.id, name="p", task="image_classification")
    s.add(project)
    await s.flush()
    s.add(Dataset(project_id=project.id, modality="vision"))
    await s.flush()
    return project


async def test_project_cleanup_covers_pool_snapshots_runs_and_pending_inference_uploads(db, calls):
    async with db() as s:
        project = await _seed_project(s)
        draft = DatasetVersion(dataset_id=project.id, version_tag=None, status="draft")
        snapshot = DatasetVersion(dataset_id=project.id, version_tag="v1", status="ready")
        s.add_all([draft, snapshot])
        s.add_all(
            [
                DatasetItem(dataset_id=project.id, storage_url="pool/p/ab/abc.png", content_hash="a" * 64),
                DatasetItem(dataset_id=project.id, storage_url="pool/p/cd/cde.png", content_hash="b" * 64),
                DatasetItem(dataset_id=project.id, storage_url=None, content_hash="c" * 64),  # inline text: no object
            ]
        )
        await s.flush()
        run = TrainingRun(project_id=project.id, dataset_version_id=snapshot.id, name="r", hyperparameters={})
        s.add(run)
        await s.flush()
        s.add(InferenceJob(run_id=run.id, payload={"kind": "file"}, upload_key="inference/i1/input.png"))
        s.add(InferenceJob(run_id=run.id, payload={"kind": "text"}, upload_key=None))
        await s.commit()
        pid, rid, sid = project.id, run.id, snapshot.id

    async with db() as s:
        await cleanup.cleanup_project_storage(s, pid)

    assert ("files", "theseus-datasets", ("pool/p/ab/abc.png", "pool/p/cd/cde.png")) in calls
    assert ("file", "theseus-datasets", f"snapshots/{sid}/dataset.parquet") in calls
    assert ("prefix", "theseus-training", f"{rid}/results/") in calls
    assert ("prefix", "theseus-training", f"{rid}/evaluation/") in calls
    assert ("prefix", "theseus-models", f"{rid}/") in calls
    assert ("files", "theseus-uploads", ("inference/i1/input.png",)) in calls
    assert ("prefix", "theseus-datasets", f"snapshots/{sid}/augmented/") in calls
    # only the real snapshot has snapshot objects (parquet, manifest, augmented copies); the draft has none
    assert sum(1 for c in calls if "snapshots/" in str(c[2])) == 3


async def test_run_cleanup_includes_that_runs_pending_uploads_only(db, calls):
    async with db() as s:
        project = await _seed_project(s)
        version = DatasetVersion(dataset_id=project.id, version_tag="v1", status="ready")
        s.add(version)
        await s.flush()
        run_a = TrainingRun(project_id=project.id, dataset_version_id=version.id, name="a", hyperparameters={})
        run_b = TrainingRun(project_id=project.id, dataset_version_id=version.id, name="b", hyperparameters={})
        s.add_all([run_a, run_b])
        await s.flush()
        s.add_all(
            [
                InferenceJob(run_id=run_a.id, payload={}, upload_key="inference/a/input.png"),
                InferenceJob(run_id=run_b.id, payload={}, upload_key="inference/b/input.png"),
            ]
        )
        await s.commit()
        run_a_id = run_a.id

    async with db() as s:
        await cleanup.cleanup_run_storage(s, run_a_id)
    assert ("files", "theseus-uploads", ("inference/a/input.png",)) in calls
    assert not any("inference/b/" in str(c) for c in calls)
