"""Training runs and sweeps over HTTP: enqueue, cancel, delete, status, evaluation, logs."""

import json
import uuid

import pytest
import sqlalchemy as sa
import yaml

from theseus.db.models import DatasetVersion, RunEvaluation, RunEvent, Sweep, TrainingMetric, TrainingRun
from theseus.events import get_event_writer
from theseus.services import snapshot as snap
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    store: dict[tuple[str, str], bytes] = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, data, content_type=None: store.__setitem__((b, k), data))
    monkeypatch.setattr(storage, "download_bytes", lambda b, k: store[(b, k)])
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "get_download_url", lambda b, k, expires_in=3600: f"https://s3.test/{b}/{k}")
    for name in ("delete_file", "delete_files", "delete_prefix"):
        monkeypatch.setattr(storage, name, lambda *a, **k: None)
    return store


async def project(c, task="image_classification"):
    r = await c.post("/api/projects", json={"name": "p", "description": None, "task": task})
    return r.json()["project"]


async def ready_version(db, fake_s3, project_id, task="image_classification", status="ready", counts=None):
    """A snapshot in the requested state, with the manifest the compiler reads back."""
    async with db() as s:
        v = DatasetVersion(dataset_id=uuid.UUID(project_id), version_tag=f"v-{uuid.uuid4().hex[:6]}", status=status)
        s.add(v)
        await s.commit()
        vid = v.id
    columns = snap.derive_columns(get_task_descriptor(task), [])
    manifest = snap.build_manifest([], ["cat", "dog"], counts or {"cat": 8, "dog": 2}, columns)
    fake_s3[("theseus-datasets", f"snapshots/{vid}/manifest.json")] = json.dumps(manifest).encode()
    return str(vid)


# -- Start training --------------------------------------------------------------------------


async def test_starting_a_run_compiles_uploads_the_config_and_queues_it(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])

    r = await c.post(
        f"/api/projects/{p['id']}/train",
        json={"name": "first", "datasetVersionId": vid, "hyperparameters": {"epochs": 7, "encoderId": "resnet50"}},
    )
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert (run["status"], run["name"], run["datasetVersionId"]) == ("queued", "first", vid)
    assert run["hyperparameters"] == {"epochs": 7, "encoderId": "resnet50"}  # stored exactly as chosen (camelCase)

    config = yaml.safe_load(fake_s3[("theseus-training", f"{run['id']}/config.yaml")])
    assert config["trainer"]["epochs"] == 7 and config["input_features"][0]["encoder"]["model_variant"] == 50
    assert config["preprocessing"]["split"]["column"] == "_ludwig_split_idx"

    await get_event_writer().flush()
    async with db() as s:
        row = (await s.execute(sa.select(TrainingRun))).scalar_one()
        events = (await s.execute(sa.select(RunEvent.payload))).scalars().all()
    assert row.config_key == f"{run['id']}/config.yaml" and row.ludwig_config["model_type"] == "ecd"
    assert [e["status"] for e in events] == ["queued"]  # the browser is told immediately over SSE


async def test_start_is_refused_for_a_missing_unready_or_foreign_version(client, new_user, db, fake_s3):
    c = await new_user()
    p, other = await project(c), await project(c)
    url = f"/api/projects/{p['id']}/train"

    assert (await c.post(url, json={"name": "x", "datasetVersionId": str(uuid.uuid4())})).status_code == 404
    building = await ready_version(db, fake_s3, p["id"], status="building")
    r = await c.post(url, json={"name": "x", "datasetVersionId": building})
    assert r.status_code == 409 and "not ready" in r.json()["error"]["message"]
    foreign = await ready_version(db, fake_s3, other["id"])
    r = await c.post(url, json={"name": "x", "datasetVersionId": foreign})
    assert r.status_code == 400 and "does not belong" in r.json()["error"]["message"]
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingRun))).scalar_one() == 0


async def test_a_config_that_cannot_compile_is_a_400_and_creates_no_run(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    r = await c.post(
        f"/api/projects/{p['id']}/train",
        json={"name": "x", "datasetVersionId": vid, "hyperparameters": {"encoderId": "nope"}},
    )
    assert r.status_code == 400 and "Failed to compile Ludwig config" in r.json()["error"]["message"]
    r = await c.post(
        f"/api/projects/{p['id']}/train",
        json={"name": "x", "datasetVersionId": vid, "hyperparameters": {"epochs": "many"}},
    )
    assert r.status_code == 422
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingRun))).scalar_one() == 0
    assert not any(k[0] == "theseus-training" for k in fake_s3)  # nothing uploaded for a rejected run


async def test_a_stranger_cannot_start_a_run_in_my_project(client, new_user, db, fake_s3):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    vid = await ready_version(db, fake_s3, p["id"])
    assert (
        await stranger.post(f"/api/projects/{p['id']}/train", json={"name": "x", "datasetVersionId": vid})
    ).status_code == 403


# -- List / detail / status ------------------------------------------------------------------


async def test_run_list_detail_and_status_shapes(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (
        await c.post(
            f"/api/projects/{p['id']}/train",
            json={"name": "r", "datasetVersionId": vid, "hyperparameters": {"epochs": 3}},
        )
    ).json()["run"]
    writer = get_event_writer()
    writer.status(run["id"], "running")
    writer.metric(run["id"], 1, "validation", {"loss": 0.9})
    writer.metric(run["id"], 2, "validation", {"loss": 0.4})
    await writer.flush()
    async with db() as s:
        s.add(RunEvaluation(run_id=uuid.UUID(run["id"]), status="success", split="test", accuracy=0.9, macro_f1=0.8))
        await s.commit()

    listed = (await c.get(f"/api/projects/{p['id']}/runs")).json()["runs"]
    assert [r["id"] for r in listed] == [run["id"]] and listed[0]["status"] == "running"
    assert listed[0]["evaluation"]["accuracy"] == pytest.approx(0.9) and listed[0]["evaluation"][
        "macroF1"
    ] == pytest.approx(0.8)

    detail = (await c.get(f"/api/runs/{run['id']}")).json()["run"]
    assert detail["datasetVersion"]["dataset"]["modality"] == "vision" and detail["datasetVersion"]["id"] == vid
    assert [(m["epoch"], m["metricValue"]) for m in detail["metrics"]] == [
        (1, pytest.approx(0.9)),
        (2, pytest.approx(0.4)),
    ]

    status = (await c.get(f"/api/runs/{run['id']}/status")).json()
    assert status["status"] == "running" and status["epochsTotal"] == 3
    assert [m["epoch"] for m in status["latestMetrics"]] == [2]  # only the most recent epoch


async def test_run_routes_are_owner_only(client, new_user, db, fake_s3):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await owner.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()[
        "run"
    ]
    for url in (
        f"/api/runs/{run['id']}",
        f"/api/runs/{run['id']}/status",
        f"/api/runs/{run['id']}/evaluation",
        f"/api/projects/{p['id']}/runs",
    ):
        assert (await stranger.get(url)).status_code == 403
    assert (await stranger.post(f"/api/runs/{run['id']}/cancel")).status_code == 403
    assert (await stranger.delete(f"/api/runs/{run['id']}")).status_code == 403


# -- Cancel / delete -------------------------------------------------------------------------


async def test_cancelling_a_queued_run_ends_it_immediately_and_it_is_never_claimed(client, new_user, db, fake_s3):
    from theseus.jobs import queue

    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]

    assert (await c.post(f"/api/runs/{run['id']}/cancel")).status_code == 204
    await get_event_writer().flush()
    assert (await c.get(f"/api/runs/{run['id']}")).json()["run"]["status"] == "canceled"
    assert await queue.claim_one(queue.TRAIN, "w", 300) is None


async def test_a_finished_run_cannot_be_cancelled_it_would_block_inference_and_export(client, new_user, db, make_run):
    c = await new_user()
    p = await project(c)
    async with db() as s:
        vid = (
            await s.execute(sa.select(DatasetVersion.id).where(DatasetVersion.dataset_id == uuid.UUID(p["id"])))
        ).scalar_one()
        run = TrainingRun(
            project_id=uuid.UUID(p["id"]), dataset_version_id=vid, name="done", status="succeeded", hyperparameters={}
        )
        s.add(run)
        await s.commit()
        rid = run.id
    r = await c.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 409 and "already finished" in r.json()["error"]["message"]
    assert (await c.get(f"/api/runs/{rid}")).json()["run"]["status"] == "succeeded"


async def test_deleting_a_run_stops_it_removes_its_rows_and_storage(client, new_user, db, fake_s3, monkeypatch):
    deleted: list = []
    monkeypatch.setattr(storage, "delete_prefix", lambda b, prefix: deleted.append((b, prefix)))
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    await get_event_writer().flush()

    assert (await c.delete(f"/api/runs/{run['id']}")).status_code == 204
    assert (await c.get(f"/api/runs/{run['id']}")).status_code == 404
    assert ("theseus-training", f"{run['id']}/results/") in deleted and ("theseus-models", f"{run['id']}/") in deleted
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingRun))).scalar_one() == 0
        assert (await s.execute(sa.select(sa.func.count()).select_from(RunEvent))).scalar_one() == 0  # events cascade


# -- Evaluation and logs ---------------------------------------------------------------------


async def _run_with_evaluation(c, db, fake_s3, top_errors):
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    async with db() as s:
        s.add(
            RunEvaluation(run_id=uuid.UUID(run["id"]), status="success", split="test", report={"topErrors": top_errors})
        )
        await s.commit()
    return p, run


async def test_evaluation_is_404_until_it_exists_then_returned_whole(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    assert (await c.get(f"/api/runs/{run['id']}/evaluation")).status_code == 404
    async with db() as s:
        s.add(
            RunEvaluation(
                run_id=uuid.UUID(run["id"]), status="success", split="full", report={"overall": {"accuracy": 1}}
            )
        )
        await s.commit()
    ev = (await c.get(f"/api/runs/{run['id']}/evaluation")).json()["evaluation"]
    assert ev["split"] == "full" and ev["report"] == {"overall": {"accuracy": 1}}


async def test_evaluation_errors_are_paginated_filtered_by_actual_class_and_joined_to_items(
    client, new_user, db, fake_s3
):
    from theseus.db.models import DatasetItem, LabelClass, TextFeatures

    c = await new_user()
    p = await project(c, "text_classification")
    vid = await ready_version(db, fake_s3, p["id"], "text_classification")
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    async with db() as s:
        cat = LabelClass(dataset_id=uuid.UUID(p["id"]), name="cat")
        s.add(cat)
        items = []
        for i in range(60):
            it = DatasetItem(
                dataset_id=uuid.UUID(p["id"]),
                content_hash=f"{i:064d}",
                storage_url=f"pool/x/{i}.txt" if i == 0 else None,
            )
            s.add(it)
            items.append(it)
        await s.flush()
        s.add(TextFeatures(item_id=items[1].id, raw_text="meow?"))
        errors = [
            {"itemId": str(it.id), "actual": "cat" if n % 2 == 0 else "dog", "predicted": "dog", "confidence": 0.5}
            for n, it in enumerate(items)
        ]
        s.add(RunEvaluation(run_id=uuid.UUID(run["id"]), status="success", split="test", report={"topErrors": errors}))
        await s.commit()
        cat_id = cat.class_id

    base = f"/api/runs/{run['id']}/evaluation/errors"
    page1 = (await c.get(base)).json()
    assert (page1["total"], page1["page"], page1["perPage"], len(page1["errors"])) == (60, 1, 50, 50)
    assert page1["errors"][0]["item"]["downloadUrl"] == "https://s3.test/theseus-datasets/pool/x/0.txt"
    assert page1["errors"][1]["item"]["text"] == "meow?" and page1["errors"][1]["item"]["downloadUrl"] is None
    assert len((await c.get(base, params={"page": 2})).json()["errors"]) == 10
    only_cats = (await c.get(base, params={"classId": str(cat_id)})).json()
    assert only_cats["total"] == 30 and {e["actual"] for e in only_cats["errors"]} == {"cat"}
    assert (await c.get(base, params={"classId": str(uuid.uuid4())})).status_code == 400
    assert (await c.get(base, params={"page": 0})).status_code == 422


async def test_evaluation_errors_are_404_without_a_successful_report(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    async with db() as s:
        s.add(RunEvaluation(run_id=uuid.UUID(run["id"]), status="failed", failed_message="boom"))
        await s.commit()
    assert (await c.get(f"/api/runs/{run['id']}/evaluation/errors")).status_code == 404


async def test_logs_redirect_to_a_presigned_url_or_404(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    run = (await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})).json()["run"]
    assert (await c.get(f"/api/runs/{run['id']}/logs")).status_code == 404
    fake_s3[("theseus-training", f"{run['id']}/logs/train.log")] = b"log"
    r = await c.get(f"/api/runs/{run['id']}/logs", follow_redirects=False)
    assert (
        r.status_code == 302 and r.headers["location"] == f"https://s3.test/theseus-training/{run['id']}/logs/train.log"
    )


# -- Sweeps ----------------------------------------------------------------------------------


async def test_a_grid_sweep_expands_into_ordered_queued_trials(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    r = await c.post(
        f"/api/projects/{p['id']}/sweeps",
        json={"name": "lr sweep", "datasetVersionId": vid, "strategy": "grid", "maxTrials": 10,
              "searchSpace": {"learningRate": [0.01, 0.001], "batchSize": [16, "auto"]}},
    )  # fmt: skip
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["sweep"]["status"] == "running" and body["sweep"]["searchSpace"] == {
        "learningRate": [0.01, 0.001],
        "batchSize": [16, "auto"],
    }
    assert [t["trialIndex"] for t in body["trials"]] == [0, 1, 2, 3]
    assert body["trials"][0]["name"] == "lr sweep — trial 1"
    assert {t["status"] for t in body["trials"]} == {"queued"}
    assert body["trials"][0]["hyperparameters"] == {"learningRate": 0.01, "batchSize": 16}

    listed = (await c.get(f"/api/projects/{p['id']}/sweeps")).json()["sweeps"]
    assert (listed[0]["trialCount"], listed[0]["completedTrialCount"]) == (4, 0)


async def test_sweep_validation_errors(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    url = f"/api/projects/{p['id']}/sweeps"
    base = {"name": "s", "datasetVersionId": vid, "strategy": "grid", "maxTrials": 4}
    assert (
        "at least one hyperparameter"
        in (await c.post(url, json={**base, "searchSpace": {}})).json()["error"]["message"]
    )
    assert (await c.post(url, json={**base, "maxTrials": 51, "searchSpace": {"epochs": [1]}})).status_code == 422
    assert (await c.post(url, json={**base, "searchSpace": {"epochs": []}})).status_code == 422
    assert (
        await c.post(url, json={**base, "datasetVersionId": str(uuid.uuid4()), "searchSpace": {"epochs": [1]}})
    ).status_code == 404
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(Sweep))).scalar_one() == 0


async def test_a_trial_that_cannot_compile_is_kept_as_a_failed_trial_not_silently_dropped(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    body = (await c.post(
        f"/api/projects/{p['id']}/sweeps",
        json={"name": "enc", "datasetVersionId": vid, "strategy": "grid", "maxTrials": 5, "searchSpace": {"encoderId": ["nope", "resnet18"]}},  # noqa: E501
    )).json()  # fmt: skip
    statuses = {t["trialIndex"]: t["status"] for t in body["trials"]}
    assert statuses == {0: "failed", 1: "queued"}
    assert "Unknown encoder" in body["trials"][0]["failedMessage"]


async def test_sweep_detail_reconciles_to_completed_once_every_trial_finished(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    body = (
        await c.post(
            f"/api/projects/{p['id']}/sweeps",
            json={
                "name": "s",
                "datasetVersionId": vid,
                "strategy": "grid",
                "maxTrials": 2,
                "searchSpace": {"epochs": [1, 2]},
            },
        )
    ).json()
    sid = body["sweep"]["id"]
    assert (await c.get(f"/api/sweeps/{sid}")).json()["sweep"]["status"] == "running"

    async with db() as s:
        await s.execute(sa.update(TrainingRun).where(TrainingRun.sweep_id == uuid.UUID(sid)).values(status="succeeded"))
        first = body["trials"][0]["id"]
        s.add(RunEvaluation(run_id=uuid.UUID(first), status="success", split="test", accuracy=0.7, macro_f1=0.6))
        await s.commit()
    detail = (await c.get(f"/api/sweeps/{sid}")).json()
    assert detail["sweep"]["status"] == "completed"
    assert [t["trialIndex"] for t in detail["trials"]] == [0, 1]
    assert (
        detail["trials"][0]["evaluation"]["accuracy"] == pytest.approx(0.7)
        and detail["trials"][1]["evaluation"] is None
    )
    assert (await c.get(f"/api/projects/{p['id']}/sweeps")).json()["sweeps"][0]["completedTrialCount"] == 2


async def test_cancelling_a_sweep_cancels_every_unfinished_trial_and_a_finished_sweep_is_409(
    client, new_user, db, fake_s3
):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    body = (
        await c.post(
            f"/api/projects/{p['id']}/sweeps",
            json={
                "name": "s",
                "datasetVersionId": vid,
                "strategy": "grid",
                "maxTrials": 3,
                "searchSpace": {"epochs": [1, 2, 3]},
            },
        )
    ).json()
    sid = body["sweep"]["id"]
    async with db() as s:  # one trial already finished on its own: it must keep its result
        await s.execute(
            sa.update(TrainingRun)
            .where(TrainingRun.id == uuid.UUID(body["trials"][0]["id"]))
            .values(status="succeeded")
        )
        await s.commit()

    assert (await c.post(f"/api/sweeps/{sid}/cancel")).status_code == 204
    await get_event_writer().flush()
    detail = (await c.get(f"/api/sweeps/{sid}")).json()
    assert detail["sweep"]["status"] == "canceled"
    assert [t["status"] for t in detail["trials"]] == ["succeeded", "canceled", "canceled"]
    assert (await c.post(f"/api/sweeps/{sid}/cancel")).status_code == 409


async def test_sweep_routes_are_owner_only(client, new_user, db, fake_s3):
    owner, stranger = await new_user(), await new_user()
    p = await project(owner)
    vid = await ready_version(db, fake_s3, p["id"])
    body = (
        await owner.post(
            f"/api/projects/{p['id']}/sweeps",
            json={
                "name": "s",
                "datasetVersionId": vid,
                "strategy": "grid",
                "maxTrials": 1,
                "searchSpace": {"epochs": [1]},
            },
        )
    ).json()
    sid = body["sweep"]["id"]
    assert (await stranger.get(f"/api/sweeps/{sid}")).status_code == 403
    assert (await stranger.post(f"/api/sweeps/{sid}/cancel")).status_code == 403
    assert (await stranger.get(f"/api/projects/{p['id']}/sweeps")).status_code == 403
    assert (
        await stranger.post(
            f"/api/projects/{p['id']}/sweeps",
            json={
                "name": "s",
                "datasetVersionId": vid,
                "strategy": "grid",
                "maxTrials": 1,
                "searchSpace": {"epochs": [1]},
            },
        )
    ).status_code == 403


async def test_metrics_table_is_untouched_by_enqueueing(client, new_user, db, fake_s3):
    c = await new_user()
    p = await project(c)
    vid = await ready_version(db, fake_s3, p["id"])
    await c.post(f"/api/projects/{p['id']}/train", json={"name": "r", "datasetVersionId": vid})
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingMetric))).scalar_one() == 0
