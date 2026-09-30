"""Bring-your-own models end to end (except a real Hub, S3 and Ludwig): the user and admin APIs, who can see
which model, training on one, the validation job, and startup recovery."""

import shutil
import uuid
import zipfile
from pathlib import Path

import pytest
import sqlalchemy as sa

from theseus.backends.registry import get_backend
from theseus.db.models import CustomModel, DatasetVersion, TrainingRun
from theseus.jobs import queue, validate_model
from theseus.jobs.recovery import recover_on_startup
from theseus.services import custom_models as cm
from theseus.services import run_models, storage
from theseus.services import training as training_service
from theseus.services.task_registry import SnapshotContext, get_task_descriptor
from theseus.services.training import QueueError, queue_training
from theseus.settings import get_settings

ADMIN_EMAIL = "boss@example.com"
TEXT = "text_classification"

HUB_BODY = {
    "name": "My BERT",
    "description": "a test model",
    "backend": "ludwig",
    "kind": "hf_transformer",
    "sourceKind": "hub",
    "sourceRef": "bert-base-uncased",
    "tasks": [TEXT],
}
UPLOAD_BODY = {**HUB_BODY, "name": "Uploaded", "sourceKind": "upload", "sourceRef": None}


# -- Fixtures and helpers ----------------------------------------------------------------------


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    return tmp_path


@pytest.fixture
async def admin(monkeypatch, new_user):
    monkeypatch.setenv("THESEUS_ADMIN_EMAILS", ADMIN_EMAIL)
    get_settings.cache_clear()
    yield await new_user(ADMIN_EMAIL)
    get_settings.cache_clear()


@pytest.fixture
def fake_storage(monkeypatch):
    """Object storage as a dict, plus a record of what was asked of it."""
    calls: dict = {"deleted_prefixes": [], "upload_urls": [], "objects": set()}
    monkeypatch.setattr(
        storage,
        "get_upload_url",
        lambda bucket, key, expires_in=3600, content_type=None: (
            calls["upload_urls"].append((key, content_type)) or f"https://s3.test/{bucket}/{key}?sig=1"
        ),
    )
    monkeypatch.setattr(storage, "file_exists", lambda bucket, key: (bucket, key) in calls["objects"])
    monkeypatch.setattr(storage, "delete_prefix", lambda bucket, prefix: calls["deleted_prefixes"].append(prefix) or 0)
    monkeypatch.setattr(storage, "upload_bytes", lambda *a, **k: None)
    return calls


async def add_model(db, **kw) -> uuid.UUID:
    fields = {
        "backend": "ludwig",
        "kind": "hf_transformer",
        "name": "Some model",
        "source_kind": "hub",
        "source_ref": "bert-base-uncased",
        "revision": "abc123",
        "tasks": [TEXT],
        "status": "ready",
        "spec": {"materialize": True},
        "max_attempts": 3,
        **kw,
    }
    async with db() as s:
        row = CustomModel(**fields)
        s.add(row)
        await s.commit()
        return row.id


async def set_status(db, model_id, **values) -> None:
    async with db() as s:
        await s.execute(sa.update(CustomModel).where(CustomModel.id == model_id).values(**values))
        await s.commit()


async def get_model(db, model_id) -> CustomModel:
    async with db() as s:
        return await s.get(CustomModel, model_id)


async def project(c, task=TEXT):
    r = await c.post("/api/projects", json={"name": "p", "description": None, "task": task})
    assert r.status_code == 200, r.text
    return r.json()["project"]


async def picker(c, project_id) -> list[dict]:
    """The models the create-run picker offers this project."""
    backends = (await c.get(f"/api/projects/{project_id}/training-backends")).json()["backends"]
    return next(b for b in backends if b["id"] == "ludwig")["models"]


def hf_bundle(path: Path, *, extra: dict[str, bytes] | None = None, causal: bool = False) -> Path:
    arch = "LlamaForCausalLM" if causal else "BertModel"
    files = {
        "my-model/config.json": f'{{"model_type": "bert", "architectures": ["{arch}"]}}'.encode(),
        "my-model/model.safetensors": b"\0" * 32,
        "my-model/tokenizer.json": b"{}",
        **{f"my-model/{k}": v for k, v in (extra or {}).items()},
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def upload_row(**kw) -> CustomModel:
    """An unsaved row shaped like a model whose bundle has been uploaded."""
    fields = {
        "id": uuid.uuid4(),
        "backend": "ludwig",
        "kind": "hf_transformer",
        "name": "x",
        "source_kind": "upload",
        "tasks": [TEXT],
        "spec": {"materialize": True},
        "storage_key": "custom/x/bundle.zip",
        **kw,
    }
    return CustomModel(**fields)


# -- Kinds -------------------------------------------------------------------------------------


async def test_kinds_list_what_can_be_added_per_task(new_user):
    user = await new_user()
    kinds = (await user.get("/api/models/kinds")).json()["kinds"]
    by_id = {k["id"]: k for k in kinds}
    assert by_id["hf_transformer"]["tasks"] == [TEXT] and by_id["hf_transformer"]["sourceKinds"] == ["hub", "upload"]
    assert "text_generation" in by_id["hf_causal_lm"]["tasks"]
    assert by_id["timm_image"]["sourceKinds"] == ["hub"] and by_id["timm_image"]["status"] == "experimental"


# -- Creating ----------------------------------------------------------------------------------


async def test_a_hub_model_is_created_private_and_goes_straight_to_validation(new_user):
    user = await new_user()
    r = await user.post("/api/models", json=HUB_BODY)
    assert r.status_code == 201, r.text
    m = r.json()["model"]
    assert m["status"] == "uploaded" and m["scope"] == "private" and m["ownerUserId"] == user.user_id
    assert m["tasks"] == [TEXT] and m["enabled"] is True and m["runCount"] == 0 and m["revision"] is None


async def test_an_upload_model_waits_for_its_file(new_user):
    user = await new_user()
    m = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    assert m["status"] == "pending_upload" and m["sourceRef"] is None


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"backend": "nope"}, "Unknown trainer backend"),
        ({"kind": "hf_causal_lm"}, "no 'hf_causal_lm' custom model"),  # not offered for text classification
        ({"tasks": ["not_a_task"]}, "Unknown task"),
        ({"tasks": [TEXT, "image_classification"]}, "no 'hf_transformer' custom model"),  # every task must fit
        ({"sourceRef": "not a repo!"}, "repository id"),
        ({"sourceRef": None}, "repository id"),
        ({"sourceKind": "upload", "sourceRef": "bert-base-uncased"}, "no source reference"),
        ({"kind": "timm_image", "tasks": ["image_classification"], "sourceKind": "upload", "sourceRef": None}, None),
    ],
)
async def test_creation_is_validated(new_user, patch, message):
    user = await new_user()
    r = await user.post("/api/models", json={**HUB_BODY, **patch})
    assert r.status_code == 400, r.text
    if message:
        assert message in r.json()["error"]["message"]


async def test_no_tasks_is_a_422(new_user):
    user = await new_user()
    assert (await user.post("/api/models", json={**HUB_BODY, "tasks": []})).status_code == 422


async def test_a_regular_user_cannot_create_a_global_model(new_user):
    user = await new_user()
    assert (await user.post("/api/admin/models", json=HUB_BODY)).status_code == 403


# -- Uploading ---------------------------------------------------------------------------------


async def test_the_upload_flow_presigns_finalizes_and_queues_validation(new_user, fake_storage):
    user = await new_user()
    m = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    base = f"/api/models/{m['id']}"

    # Finalizing before an upload URL was requested, or before the object exists, is refused.
    assert (await user.post(f"{base}/finalize")).status_code == 409

    r = await user.post(f"{base}/upload-url", json={"filename": "Model.ZIP", "sizeBytes": 1000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"] == f"custom/{m['id']}/bundle.zip" and body["method"] == "PUT"
    assert body["headers"] == {"Content-Type": "application/zip"} and body["url"].startswith("https://s3.test/")
    assert fake_storage["upload_urls"] == [(body["key"], "application/zip")]

    assert (await user.post(f"{base}/finalize")).status_code == 409  # signed, but nothing uploaded yet
    fake_storage["objects"].add(("theseus-models", body["key"]))
    r = await user.post(f"{base}/finalize")
    assert r.status_code == 200 and r.json()["model"]["status"] == "uploaded"
    assert (await user.post(f"{base}/finalize")).status_code == 409  # only once


async def test_a_single_safetensors_file_is_accepted(new_user, fake_storage):
    user = await new_user()
    m = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    r = await user.post(f"/api/models/{m['id']}/upload-url", json={"filename": "w.safetensors", "sizeBytes": 5})
    assert r.json()["key"].endswith("/model.safetensors")
    assert r.json()["headers"] == {"Content-Type": "application/octet-stream"}


async def test_upload_urls_refuse_bad_files_sizes_and_states(new_user, fake_storage, monkeypatch):
    user = await new_user()
    upload = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    hub = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    url = "/api/models/{}/upload-url"

    assert (await user.post(url.format(upload["id"]), json={"filename": "w.bin", "sizeBytes": 5})).status_code == 400
    assert (await user.post(url.format(upload["id"]), json={"filename": "w.pt", "sizeBytes": 5})).status_code == 400
    monkeypatch.setattr(get_settings(), "max_custom_model_bytes", 100)
    assert (await user.post(url.format(upload["id"]), json={"filename": "w.zip", "sizeBytes": 101})).status_code == 413
    assert (await user.post(url.format(upload["id"]), json={"filename": "w.zip", "sizeBytes": 0})).status_code == 422
    assert (await user.post(url.format(hub["id"]), json={"filename": "w.zip", "sizeBytes": 5})).status_code == 409


# -- Who can see and change what ---------------------------------------------------------------


async def test_a_model_is_invisible_to_everyone_but_its_owner(new_user, fake_storage):
    owner, other = await new_user(), await new_user()
    m = (await owner.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    base = f"/api/models/{m['id']}"

    assert [x["id"] for x in (await owner.get("/api/models")).json()["models"]] == [m["id"]]
    assert (await other.get("/api/models")).json()["models"] == []
    for call in (
        other.patch(base, json={"name": "hijack"}),
        other.delete(base),
        other.post(f"{base}/upload-url", json={"filename": "a.zip", "sizeBytes": 1}),
        other.post(f"{base}/finalize"),
        other.post(f"{base}/retry"),
    ):
        assert (await call).status_code == 404  # not 403: an id must not reveal that someone's model exists


async def test_a_model_can_be_renamed_retasked_and_switched_off(new_user):
    user = await new_user()
    m = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    r = await user.patch(f"/api/models/{m['id']}", json={"name": " Renamed ", "description": "d", "enabled": False})
    got = r.json()["model"]
    assert got["name"] == "Renamed" and got["description"] == "d" and got["enabled"] is False
    # What the model IS is fixed, and a task its kind does not support cannot be added.
    bad = await user.patch(f"/api/models/{m['id']}", json={"tasks": [TEXT, "image_classification"]})
    assert bad.status_code == 400
    assert (await user.patch(f"/api/models/{m['id']}", json={"tasks": []})).status_code == 422


async def test_deleting_an_unused_model_removes_it_and_its_files(new_user, fake_storage, temp_dir):
    user = await new_user()
    m = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]
    leftover = temp_dir / "custom_models" / m["id"] / "pin"
    leftover.mkdir(parents=True)
    (leftover / "f").write_text("x")

    r = await user.delete(f"/api/models/{m['id']}")
    assert r.json() == {"deleted": True, "archived": False}
    assert fake_storage["deleted_prefixes"] == [f"custom/{m['id']}/"]
    assert not leftover.exists()
    assert (await user.get("/api/models")).json()["models"] == []


async def test_a_model_a_run_trained_on_is_archived_not_deleted(new_user, fake_storage, db, make_run):
    user = await new_user()
    m = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    run_id = await make_run()
    async with db() as s:
        await s.execute(
            sa.update(TrainingRun).where(TrainingRun.id == run_id).values(custom_model_id=uuid.UUID(m["id"]))
        )
        await s.commit()

    r = await user.delete(f"/api/models/{m['id']}")
    assert r.json() == {"deleted": False, "archived": True}
    assert fake_storage["deleted_prefixes"] == []  # files stay so the run can still be served and exported
    row = await get_model(db, uuid.UUID(m["id"]))
    assert row.archived_at is not None and row.enabled is False
    assert (await user.get("/api/models")).json()["models"] == []  # hidden from the list
    assert (await user.patch(f"/api/models/{m['id']}", json={"name": "x"})).status_code == 409


async def test_deleting_succeeds_even_when_the_object_store_cannot_be_reached(new_user, fake_storage, monkeypatch):
    def unreachable(bucket, prefix):
        raise ConnectionError("object store is down")

    monkeypatch.setattr(storage, "delete_prefix", unreachable)
    user = await new_user()
    m = (await user.post("/api/models", json=UPLOAD_BODY)).json()["model"]

    r = await user.delete(f"/api/models/{m['id']}")
    assert r.status_code == 200 and r.json() == {"deleted": True, "archived": False}  # the row really is gone
    assert (await user.get("/api/models")).json()["models"] == []


async def test_a_model_being_validated_cannot_be_deleted(new_user, db):
    user = await new_user()
    m = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    await set_status(db, uuid.UUID(m["id"]), status="validating")
    assert (await user.delete(f"/api/models/{m['id']}")).status_code == 409


async def test_only_a_failed_model_can_be_retried(new_user, db):
    user = await new_user()
    m = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    assert (await user.post(f"/api/models/{m['id']}/retry")).status_code == 409
    await set_status(db, uuid.UUID(m["id"]), status="failed", last_error="boom", attempt=3)
    r = await user.post(f"/api/models/{m['id']}/retry")
    assert r.status_code == 200 and r.json()["model"]["status"] == "uploaded" and r.json()["model"]["lastError"] is None
    assert (await get_model(db, uuid.UUID(m["id"]))).attempt == 0


# -- Admin -------------------------------------------------------------------------------------


async def test_an_admin_creates_a_global_model_and_sees_everyones(admin, new_user):
    user = await new_user("alice@example.com")
    mine = (await user.post("/api/models", json={**HUB_BODY, "name": "Alice's"})).json()["model"]
    made = await admin.post("/api/admin/models", json={**HUB_BODY, "name": "Shared"})
    assert made.status_code == 201
    shared = made.json()["model"]
    assert shared["scope"] == "global" and shared["ownerUserId"] is None

    everything = (await admin.get("/api/admin/models")).json()["models"]
    assert {m["name"] for m in everything} == {"Alice's", "Shared"}
    assert next(m for m in everything if m["id"] == mine["id"])["ownerEmail"] == "alice@example.com"
    assert [m["name"] for m in (await admin.get("/api/admin/models", params={"scope": "global"})).json()["models"]] == [
        "Shared"
    ]
    assert [
        m["name"] for m in (await admin.get("/api/admin/models", params={"scope": "private"})).json()["models"]
    ] == ["Alice's"]
    assert [m["name"] for m in (await admin.get("/api/admin/models", params={"q": "shar"})).json()["models"]] == [
        "Shared"
    ]
    assert (await admin.get("/api/admin/models", params={"status": "ready"})).json()["models"] == []


async def test_only_admins_reach_the_admin_model_api(new_user):
    user = await new_user()
    assert (await user.get("/api/admin/models")).status_code == 403


async def test_an_admin_can_manage_a_users_private_model(admin, new_user, fake_storage):
    user = await new_user()
    m = (await user.post("/api/models", json=HUB_BODY)).json()["model"]
    r = await admin.patch(f"/api/admin/models/{m['id']}", json={"enabled": False})
    assert r.status_code == 200 and r.json()["model"]["enabled"] is False
    assert (await admin.delete(f"/api/admin/models/{m['id']}")).json() == {"deleted": True, "archived": False}
    assert (await user.get("/api/models")).json()["models"] == []


# -- Which models a project's picker offers ----------------------------------------------------


async def test_the_picker_lists_builtins_then_the_custom_models_the_owner_may_use(admin, new_user, db):
    owner, other = await new_user(), await new_user()
    global_id = await add_model(db, name="Global BERT")
    mine = await add_model(db, name="Mine", owner_user_id=uuid.UUID(owner.user_id))
    theirs = await add_model(db, name="Theirs", owner_user_id=uuid.UUID(other.user_id))
    p = await project(owner)

    models = await picker(owner, p["id"])
    ids = [m["id"] for m in models]
    assert ids[:4] == ["bert", "distilbert", "roberta", "stacked_cnn"]  # built-ins keep their place
    assert ids[4:] == [f"custom:{global_id}", f"custom:{mine}"]
    assert f"custom:{theirs}" not in ids
    custom = {m["id"]: m for m in models if m["id"].startswith("custom:")}
    assert custom[f"custom:{global_id}"]["source"] == "global" and custom[f"custom:{mine}"]["source"] == "private"
    assert custom[f"custom:{mine}"]["kind"] == "hf_transformer" and custom[f"custom:{mine}"]["pretrained"] is True
    assert all(m["source"] == "builtin" and m["kind"] is None for m in models if m["id"] in ids[:4])


@pytest.mark.parametrize(
    "hidden",
    [
        {"status": "uploaded"},
        {"status": "validating"},
        {"status": "failed"},
        {"status": "pending_upload"},
        {"enabled": False},
        {"archived_at": sa.func.now()},
        {"tasks": ["image_classification"]},
        {"backend": "other"},
        {"kind": "hf_causal_lm"},  # a kind the task no longer offers
    ],
)
async def test_a_custom_model_that_is_not_usable_is_not_offered(new_user, db, hidden):
    user = await new_user()
    await add_model(db, **hidden)
    assert [m["id"] for m in await picker(user, (await project(user))["id"]) if m["id"].startswith("custom:")] == []


async def test_other_tasks_do_not_see_the_model(new_user, db):
    user = await new_user()
    await add_model(db)
    other_task = await project(user, "tabular_classification")
    backends = (await user.get(f"/api/projects/{other_task['id']}/training-backends")).json()["backends"]
    assert next(b for b in backends if b["id"] == "ludwig")["models"] == []


# -- Training on a custom model ----------------------------------------------------------------


@pytest.fixture
def training_env(monkeypatch, fake_storage, temp_dir):
    async def manifest(_version_id):
        return SnapshotContext(label_class_names=["a", "b"])

    monkeypatch.setattr(training_service, "read_snapshot_manifest", manifest)


async def ready_version(db, project_id) -> uuid.UUID:
    async with db() as s:
        v = DatasetVersion(dataset_id=uuid.UUID(project_id), version_tag="v1", status="ready")
        s.add(v)
        await s.commit()
        return v.id


async def queue_run(db, project_id, version_id, **hp):
    async with db() as s:
        return await queue_training(
            s, project_id=uuid.UUID(project_id), name="r", task=TEXT, dataset_version_id=version_id, hyperparameters=hp
        )


async def test_a_run_on_a_custom_model_records_it_and_compiles_its_path(new_user, db, training_env):
    user = await new_user()
    model_id = await add_model(db, owner_user_id=uuid.UUID(user.user_id))
    p = await project(user)
    run = await queue_run(db, p["id"], await ready_version(db, p["id"]), encoderId=f"custom:{model_id}")

    assert isinstance(run, TrainingRun), run
    assert run.custom_model_id == model_id
    assert run.hyperparameters["encoderId"] == f"custom:{model_id}"
    encoder = run.config["input_features"][0]["encoder"]
    assert encoder["type"] == "auto_transformer"
    assert encoder["pretrained_model_name_or_path"] == str(cm.local_dir_for(await get_model(db, model_id)))


async def test_a_run_on_a_builtin_model_records_no_custom_model(new_user, db, training_env):
    user = await new_user()
    p = await project(user)
    run = await queue_run(db, p["id"], await ready_version(db, p["id"]), encoderId="bert")
    assert isinstance(run, TrainingRun) and run.custom_model_id is None


@pytest.mark.parametrize("change", [{"enabled": False}, {"status": "failed"}, {"tasks": ["image_classification"]}])
async def test_training_refuses_a_custom_model_that_is_no_longer_usable(new_user, db, training_env, change):
    user = await new_user()
    model_id = await add_model(db, owner_user_id=uuid.UUID(user.user_id), **change)
    p = await project(user)
    result = await queue_run(db, p["id"], await ready_version(db, p["id"]), encoderId=f"custom:{model_id}")
    assert isinstance(result, QueueError) and result.code == 400 and "not available" in result.message


async def test_training_cannot_use_someone_elses_private_model_or_a_made_up_id(new_user, db, training_env):
    user, other = await new_user(), await new_user()
    theirs = await add_model(db, owner_user_id=uuid.UUID(other.user_id))
    p = await project(user)
    version = await ready_version(db, p["id"])
    for bad in (f"custom:{theirs}", f"custom:{uuid.uuid4()}", "custom:not-a-uuid"):
        result = await queue_run(db, p["id"], version, encoderId=bad)
        assert isinstance(result, QueueError) and result.code == 400, bad
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(TrainingRun))).scalar_one() == 0


async def test_a_global_model_is_usable_by_anyone(new_user, db, training_env):
    user = await new_user()
    model_id = await add_model(db)  # no owner
    p = await project(user)
    run = await queue_run(db, p["id"], await ready_version(db, p["id"]), encoderId=f"custom:{model_id}")
    assert isinstance(run, TrainingRun) and run.custom_model_id == model_id


async def test_a_causal_lm_is_the_base_model_of_an_llm_task(new_user, db, monkeypatch, fake_storage, temp_dir):
    async def manifest(_):
        return SnapshotContext()

    monkeypatch.setattr(training_service, "read_snapshot_manifest", manifest)
    user = await new_user()
    model_id = await add_model(db, kind="hf_causal_lm", tasks=["text_generation"], source_ref="org/llama")
    p = await project(user, "text_generation")
    async with db() as s:
        v = DatasetVersion(dataset_id=uuid.UUID(p["id"]), version_tag="v1", status="ready")
        s.add(v)
        await s.commit()
        run = await queue_training(
            s, project_id=uuid.UUID(p["id"]), name="r", task="text_generation", dataset_version_id=v.id,
            hyperparameters={"encoderId": f"custom:{model_id}"},
        )  # fmt: skip
    assert isinstance(run, TrainingRun), run
    assert run.config["model_type"] == "llm"
    assert run.config["base_model"] == str(cm.local_dir_for(await get_model(db, model_id)))


# -- The validation job ------------------------------------------------------------------------


@pytest.fixture
def bundle_download(monkeypatch, tmp_path):
    """`storage.download_file` serves a prepared local file for whatever key is asked."""
    served: dict[str, Path] = {}

    def download(bucket, key, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(served["file"], local_path)
        return local_path

    monkeypatch.setattr(storage, "download_file", download)
    return served


def test_validating_an_upload_pins_its_hash_and_unpacks_it_where_ensure_local_finds_it(temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip")
    row = upload_row()
    result = validate_model.validate_sync(row)

    assert result.sha256 == cm.sha256_file(temp_dir / "b.zip") and result.revision is None
    assert result.size_bytes and row.sha256 == result.sha256
    path = cm.local_dir_for(row)
    assert path.name == result.sha256 and (path / "config.json").is_file() and (path / "model.safetensors").is_file()
    assert cm.ensure_local(row) == path  # already in place: no second download
    assert cm.ref_for(row).local_path == str(path)


def test_a_bundle_with_pickled_weights_fails_with_a_message_the_user_can_act_on(temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip", extra={"pytorch_model.bin": b"x"})
    with pytest.raises(cm.CustomModelFileError, match="convert the weights to .safetensors"):
        validate_model.validate_sync(upload_row())


def test_the_backends_own_check_runs_on_every_task(temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip")  # a BertModel, not a causal LM
    row = upload_row(kind="hf_causal_lm")
    row.tasks = ["text_generation"]
    with pytest.raises(Exception, match="Not a causal language model"):
        validate_model.validate_sync(row)


def test_a_model_offered_for_no_task_or_an_unoffered_kind_is_refused(temp_dir):
    row = upload_row()
    row.tasks = []
    with pytest.raises(Exception, match="does not offer"):
        validate_model.validate_sync(row)
    row = upload_row(kind="hf_causal_lm")  # not offered for text classification
    with pytest.raises(Exception, match="does not offer"):
        validate_model.validate_sync(row)


def hub_row(**kw) -> CustomModel:
    fields = {
        "id": uuid.uuid4(),
        "backend": "ludwig",
        "kind": "hf_transformer",
        "name": "x",
        "source_kind": "hub",
        "source_ref": "org/bert",
        "tasks": [TEXT],
        "spec": {"materialize": True},
        **kw,
    }
    return CustomModel(**fields)


def fake_hub(monkeypatch, *, sha="c0ffee", size=100):
    monkeypatch.setattr(validate_model, "_hub_pin", lambda row: (sha, size))

    def fetch(row, dest: Path):
        assert row.revision == sha  # pinned before anything is downloaded
        (dest / "config.json").write_text('{"model_type": "bert", "architectures": ["BertModel"]}')
        (dest / "model.safetensors").write_bytes(b"\0" * 8)

    monkeypatch.setattr(cm, "_fetch_hub", fetch)


def test_validating_a_hub_model_pins_the_commit_and_fetches_at_that_revision(temp_dir, monkeypatch):
    fake_hub(monkeypatch)
    row = hub_row()
    result = validate_model.validate_sync(row)
    assert (
        result.revision == "c0ffee"
        and result.sha256 is None
        and result.size_bytes == 8 + len('{"model_type": "bert", "architectures": ["BertModel"]}')
    )
    assert cm.local_dir_for(row).name == "c0ffee"


def test_a_hub_model_over_the_size_limit_is_refused_before_downloading(temp_dir, monkeypatch):
    fake_hub(monkeypatch, size=10 * 2**20)
    monkeypatch.setattr(get_settings(), "max_custom_model_bytes", 2**20)
    with pytest.raises(cm.CustomModelFileError, match="over the"):
        validate_model.validate_sync(hub_row())
    assert not (temp_dir / "custom_models").exists()


def test_a_stored_bundle_that_changed_is_caught_when_it_is_put_back(temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip")
    row = upload_row()
    validate_model.validate_sync(row)
    shutil.rmtree(cm.local_dir_for(row))  # a restart wiped the temp directory

    bundle_download["file"] = hf_bundle(temp_dir / "b2.zip", extra={"extra.txt": b"tampered"})
    with pytest.raises(cm.CustomModelFileError, match="no longer matches"):
        cm.ensure_local(row)


def test_a_kind_that_fetches_its_own_weights_has_nothing_to_materialise(temp_dir):
    row = hub_row(kind="timm_image", source_ref="resnet50", revision=None)
    row.spec = {"materialize": False}
    assert cm.ensure_local(row) == cm.local_dir_for(row) and not cm.local_dir_for(row).exists()


async def claim(db, model_id) -> None:
    """Move a model to `validating` the way the dispatcher does."""
    claimed = await queue.claim_one(queue.VALIDATE, "worker-1", 60)
    assert claimed == model_id


async def test_the_job_marks_a_good_upload_ready(db, temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip")
    model_id = await add_model(
        db, status="uploaded", source_kind="upload", source_ref=None, revision=None,
        storage_key="custom/x/bundle.zip",
    )  # fmt: skip
    await claim(db, model_id)
    await validate_model.run_validate(model_id)

    row = await get_model(db, model_id)
    assert row.status == "ready" and row.last_error is None and row.claimed_by is None
    assert row.sha256 == cm.sha256_file(temp_dir / "b.zip") and row.size_bytes > 0


async def test_the_job_fails_a_bad_model_at_once_with_the_reason(db, temp_dir, bundle_download):
    bundle_download["file"] = hf_bundle(temp_dir / "b.zip", extra={"weights.pt": b"x"})
    model_id = await add_model(
        db, status="uploaded", source_kind="upload", source_ref=None, revision=None,
        storage_key="custom/x/bundle.zip",
    )  # fmt: skip
    await claim(db, model_id)
    await validate_model.run_validate(model_id)

    row = await get_model(db, model_id)
    assert row.status == "failed" and ".safetensors" in row.last_error
    assert row.attempt == 1  # a deterministic failure is not retried
    assert not (temp_dir / "custom_models" / str(model_id)).exists()  # no half-unpacked leftovers


async def test_an_unexpected_error_is_retried_then_fails_when_attempts_run_out(db):
    model_id = await add_model(db, status="uploaded", max_attempts=2)
    await claim(db, model_id)
    await validate_model.handle_failure(model_id, ConnectionError("hub down"))
    row = await get_model(db, model_id)
    assert row.status == "uploaded" and "hub down" in row.last_error  # back in the queue, after a delay

    await set_status(db, model_id, available_at=sa.func.now())
    await claim(db, model_id)
    await validate_model.handle_failure(model_id, ConnectionError("hub still down"))
    assert (await get_model(db, model_id)).status == "failed"


async def test_a_model_deleted_while_queued_is_ignored_by_the_job(db):
    await validate_model.run_validate(uuid.uuid4())  # no row: logs and returns


async def test_startup_requeues_an_interrupted_validation_or_fails_it_once_out_of_attempts(app, db):
    retry = await add_model(db, status="validating", attempt=1, max_attempts=3, claimed_by="old")
    spent = await add_model(db, status="validating", attempt=3, max_attempts=3, claimed_by="old")
    report = await recover_on_startup()

    assert (report.models_requeued, report.models_failed) == (1, 1)
    assert (await get_model(db, retry)).status == "uploaded" and (await get_model(db, retry)).claimed_by is None
    failed = await get_model(db, spent)
    assert failed.status == "failed" and "restarted" in failed.last_error


async def test_the_dispatcher_has_a_validate_lane():
    from theseus.jobs.dispatcher import build_default_lanes

    lanes = {lane.name: lane for lane in build_default_lanes()}
    assert (
        lanes["validate"].kind is queue.VALIDATE
        and lanes["validate"].concurrency == 1
        and lanes["validate"].renew_lease
    )


# -- Loading a trained model ------------------------------------------------------------------


async def test_a_run_on_a_custom_model_gets_its_files_back_before_loading(db, make_run, monkeypatch):
    model_id = await add_model(db)
    with_model, without = await make_run(), await make_run()
    async with db() as s:
        await s.execute(sa.update(TrainingRun).where(TrainingRun.id == with_model).values(custom_model_id=model_id))
        await s.commit()

    ensured: list[uuid.UUID] = []
    monkeypatch.setattr(cm, "ensure_local", lambda row: ensured.append(row.id))
    await run_models.ensure_run_custom_model(without)  # a built-in run: nothing to do
    assert ensured == []
    await run_models.ensure_run_custom_model(str(with_model))  # accepts the string form the cache uses
    assert ensured == [model_id]


async def test_the_backend_contract_defaults_offer_no_custom_models():
    from theseus.backends.base import TrainerBackend

    assert TrainerBackend.custom_model_kinds.__func__(TrainerBackend, get_task_descriptor(TEXT)) == []
    assert get_backend("ludwig").custom_model_kinds(get_task_descriptor("tabular_classification")) == []


# -- Naming the model a run used ---------------------------------------------------------------


async def test_run_detail_names_the_model_it_used(new_user, db, make_run):
    user = await new_user()
    project_id = uuid.UUID((await project(user))["id"])
    model_id = await add_model(db, name="Legal BERT", owner_user_id=uuid.UUID(user.user_id))
    async with db() as s:
        version = DatasetVersion(dataset_id=project_id, version_tag="v1", status="ready")
        s.add(version)
        await s.flush()

        def run(**kw):
            return TrainingRun(
                project_id=project_id, dataset_version_id=version.id, name="r", status="succeeded",
                hyperparameters=kw.pop("hyperparameters", {}), config_key="cfg", **kw,
            )  # fmt: skip

        custom, builtin, unnamed = (
            run(hyperparameters={"encoderId": f"custom:{model_id}"}, custom_model_id=model_id),
            run(hyperparameters={"encoderId": "distilbert"}),
            run(),
        )
        s.add_all([custom, builtin, unnamed])
        await s.commit()
        ids = (custom.id, builtin.id, unnamed.id)

    async def detail(run_id):
        return (await user.get(f"/api/runs/{run_id}")).json()["run"]

    got = await detail(ids[0])
    assert got["modelLabel"] == "Legal BERT" and got["customModelId"] == str(model_id) and got["backend"] == "ludwig"
    assert (await detail(ids[1]))["modelLabel"] == "DistilBERT"
    assert (await detail(ids[2]))["modelLabel"] is None

    # Archiving the model (it is in use, so deleting archives it) does not lose the name a run shows.
    assert (await user.delete(f"/api/models/{model_id}")).json()["archived"] is True
    assert (await detail(ids[0]))["modelLabel"] == "Legal BERT"


async def test_a_string_that_is_not_a_run_id_has_no_custom_model_and_touches_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not query or materialise for something that is not a run id")

    monkeypatch.setattr(cm, "ensure_local", boom)
    monkeypatch.setattr(run_models, "get_sessionmaker", boom)
    await run_models.ensure_run_custom_model("run-a")
