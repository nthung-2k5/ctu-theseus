"""Admin enable/disable switches: stored, mirrored in-process, and applied to listings AND validation."""

import uuid

import pytest
import sqlalchemy as sa

from theseus.backends.registry import get_backend, trainable_backends
from theseus.db.models import DatasetVersion, PluginSetting, TrainingRun
from theseus.export.registry import find_export_format, list_export_formats
from theseus.preprocessing.config import PreprocessingConfig
from theseus.preprocessing.registry import PreprocessingConfigError, list_preprocessing, validate_config
from theseus.services import plugin_settings as ps
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor
from theseus.services.training import QueueError, _check_builtin_model, queue_training
from theseus.settings import get_settings

ADMIN_EMAIL = "boss@example.com"
TASK = "image_classification"


@pytest.fixture
async def admin(monkeypatch, new_user):
    monkeypatch.setenv("THESEUS_ADMIN_EMAILS", ADMIN_EMAIL)
    get_settings.cache_clear()
    c = await new_user(ADMIN_EMAIL)
    yield c
    get_settings.cache_clear()


async def put(admin, kind, plugin_id, enabled, task=None):
    body = {"enabled": enabled, **({"task": task} if task else {})}
    return await admin.put(f"/api/admin/plugins/{kind}/{plugin_id}", json=body)


async def create_project(c, task=TASK):
    r = await c.post("/api/projects", json={"name": "p", "description": None, "task": task})
    return r


async def count(db, model) -> int:
    async with db() as s:
        return (await s.execute(sa.select(sa.func.count()).select_from(model))).scalar_one()


# -- Semantics (no HTTP) -----------------------------------------------------------------------


def test_nothing_is_disabled_by_default():
    assert ps.is_enabled("export_format", "onnx")
    assert ps.is_enabled("export_format", "onnx", TASK)


def test_a_task_override_beats_the_every_task_setting(monkeypatch):
    monkeypatch.setattr(ps, "_state", {("export_format", "onnx", ""): False, ("export_format", "onnx", TASK): True})
    assert not ps.is_enabled("export_format", "onnx")
    assert not ps.is_enabled("export_format", "onnx", "text_classification")
    assert ps.is_enabled("export_format", "onnx", TASK)


# -- Admin API ---------------------------------------------------------------------------------


async def test_only_admins_can_list_or_change_plugins(new_user):
    user = await new_user()
    assert (await user.get("/api/admin/plugins")).status_code == 403
    assert (await put(user, "export_format", "onnx", False)).status_code == 403


async def test_listing_covers_every_kind(admin):
    plugins = (await admin.get("/api/admin/plugins")).json()["plugins"]
    kinds = {p["kind"] for p in plugins}
    assert kinds == {"backend", "builtin_model", "export_format", "preprocessing", "augmentation", "task"}
    resnet = next(p for p in plugins if p["id"] == "ludwig:resnet18")
    assert resnet["kind"] == "builtin_model" and resnet["group"] == "ludwig"
    assert {t["task"] for t in resnet["tasks"]} == {"image_classification", "image_captioning"}
    assert all(p["enabled"] for p in plugins)


async def test_unknown_kind_plugin_and_task_are_rejected(admin):
    assert (await put(admin, "nonsense", "onnx", False)).status_code == 404
    assert (await put(admin, "export_format", "no_such_format", False)).status_code == 404
    assert (await put(admin, "export_format", "onnx", False, task="not_a_task")).status_code == 400


async def test_setting_is_persisted_reflected_and_clearable(admin, db):
    r = await put(admin, "export_format", "onnx", False)
    assert r.status_code == 200 and r.json()["plugin"]["enabled"] is False

    async with db() as s:
        row = (await s.execute(sa.select(PluginSetting))).scalar_one()
        assert (row.kind, row.plugin_id, row.task, row.enabled) == ("export_format", "onnx", "", False)

    # Writing again updates in place rather than adding a second row.
    await put(admin, "export_format", "onnx", False)
    assert await count(db, PluginSetting) == 1

    r = await put(admin, "export_format", "onnx", None)  # clear: inherit the default again
    assert r.json()["plugin"]["enabled"] is True
    assert await count(db, PluginSetting) == 0


async def test_per_task_state_is_reported_with_overrides(admin):
    await put(admin, "export_format", "onnx", False)
    r = await put(admin, "export_format", "onnx", True, task=TASK)
    plugin = r.json()["plugin"]
    assert plugin["enabled"] is False
    by_task = {t["task"]: t for t in plugin["tasks"]}
    assert by_task[TASK]["enabled"] is True and by_task[TASK]["overridden"] is True
    other = next(t for t in plugin["tasks"] if t["task"] != TASK)
    assert other["enabled"] is False and other["overridden"] is False


async def test_settings_are_loaded_from_the_database_at_startup(admin):
    await put(admin, "export_format", "onnx", False)
    ps.reset()
    assert ps.is_enabled("export_format", "onnx")
    await ps.load()
    assert not ps.is_enabled("export_format", "onnx")


# -- Enforcement: listings and validation ------------------------------------------------------


async def test_a_disabled_export_format_is_hidden_and_rejected_but_still_findable(admin, db):
    await put(admin, "export_format", "onnx", False)
    assert "onnx" not in {f["id"] for f in (await admin.get("/api/export-formats")).json()["formats"]}
    assert "onnx" in {f.id for f in list_export_formats(include_disabled=True)}
    # An export queued before the switch still resolves its format, so it can run.
    assert find_export_format("onnx") is not None

    project = (await create_project(admin)).json()["project"]
    async with db() as s:
        v = DatasetVersion(dataset_id=uuid.UUID(project["id"]), version_tag="v1", status="ready")
        s.add(v)
        await s.flush()
        run = TrainingRun(
            project_id=uuid.UUID(project["id"]), dataset_version_id=v.id, name="r", status="succeeded",
            hyperparameters={}, config_key="cfg",
        )  # fmt: skip
        s.add(run)
        await s.commit()
        run_id = run.id
    r = await admin.post(f"/api/runs/{run_id}/exports", json={"format": "onnx"})
    assert r.status_code == 400 and "disabled" in r.json()["error"]["message"]


def test_a_disabled_preprocessing_op_is_hidden_and_fails_validation(monkeypatch):
    task = get_task_descriptor(TASK)
    op = list_preprocessing(task)[0]
    monkeypatch.setattr(ps, "_state", {("preprocessing", op.id, ""): False})
    assert op.id not in {o.id for o in list_preprocessing(task)}
    assert op.id in {o.id for o in list_preprocessing(task, include_disabled=True)}
    config = PreprocessingConfig.model_validate({"ops": [{"id": op.id, "params": {}, "splits": ["train"]}]})
    with pytest.raises(PreprocessingConfigError, match="disabled by an administrator"):
        validate_config(task, config)


async def test_a_disabled_task_blocks_new_projects_only(admin):
    existing = (await create_project(admin)).json()["project"]
    await put(admin, "task", TASK, False)

    r = await create_project(admin)
    assert r.status_code == 422 and "disabled" in r.json()["error"]["message"]
    assert (await admin.get(f"/api/projects/{existing['id']}")).status_code == 200  # existing project untouched

    ludwig = next(b for b in (await admin.get("/api/training-backends")).json()["backends"] if b["id"] == "ludwig")
    assert TASK not in ludwig["supportedTasks"] and "text_classification" in ludwig["supportedTasks"]


async def test_a_disabled_backend_is_not_trainable_for_that_task_only(admin):
    ludwig = get_backend("ludwig")
    await put(admin, "backend", "ludwig", False, task=TASK)
    assert ludwig not in trainable_backends(get_task_descriptor(TASK))
    assert ludwig in trainable_backends(get_task_descriptor("text_classification"))
    assert (await create_project(admin)).status_code == 422  # nothing can train it now


async def test_a_disabled_builtin_model_leaves_the_picker(admin):
    await put(admin, "builtin_model", "ludwig:resnet18", False)
    project = (await create_project(admin)).json()["project"]
    ludwig = (await admin.get(f"/api/projects/{project['id']}/training-backends")).json()["backends"][0]
    ids = [m["id"] for m in ludwig["models"]]
    assert "resnet18" not in ids and "resnet50" in ids
    assert "resnet18" in [m.id for m in get_backend("ludwig").models(get_task_descriptor(TASK))]  # still installed


# -- Enforcement: queueing a run ---------------------------------------------------------------


class _Sel:
    def __init__(self, model_id=None):
        self.model_id = model_id


def test_default_model_skips_a_disabled_first_choice(monkeypatch):
    monkeypatch.setattr(ps, "_state", {("builtin_model", "ludwig:resnet18", ""): False})
    sel = _Sel()
    assert _check_builtin_model(get_backend("ludwig"), TASK, sel) is None
    assert sel.model_id == "resnet50"  # the next enabled model, named explicitly


def test_default_model_is_left_unset_when_the_first_choice_is_enabled():
    sel = _Sel()
    assert _check_builtin_model(get_backend("ludwig"), TASK, sel) is None
    assert sel.model_id is None  # nothing changes for an install with no overrides


def test_an_explicitly_disabled_model_is_refused(monkeypatch):
    monkeypatch.setattr(ps, "_state", {("builtin_model", "ludwig:resnet50", ""): False})
    assert "disabled" in _check_builtin_model(get_backend("ludwig"), TASK, _Sel("resnet50"))
    assert _check_builtin_model(get_backend("ludwig"), TASK, _Sel("resnet18")) is None


def test_no_model_left_is_an_error(monkeypatch):
    ludwig = get_backend("ludwig")
    models = ludwig.models(get_task_descriptor(TASK))
    monkeypatch.setattr(ps, "_state", {("builtin_model", f"ludwig:{m.id}", ""): False for m in models})
    assert "Every model" in _check_builtin_model(ludwig, TASK, _Sel())


async def test_queueing_refuses_a_disabled_backend_and_model(admin, db, monkeypatch):
    monkeypatch.setattr(storage, "upload_bytes", lambda *a, **k: None)
    project_id = uuid.UUID((await create_project(admin)).json()["project"]["id"])
    async with db() as s:
        v = DatasetVersion(dataset_id=project_id, version_tag="v1", status="ready")
        s.add(v)
        await s.commit()
        version_id = v.id

    await put(admin, "builtin_model", "ludwig:resnet50", False)
    async with db() as s:
        err = await queue_training(
            s, project_id=project_id, name="r", task=TASK, dataset_version_id=version_id,
            hyperparameters={"encoderId": "resnet50"},
        )  # fmt: skip
        assert isinstance(err, QueueError) and err.code == 400 and "disabled" in err.message

    await put(admin, "backend", "ludwig", False, task=TASK)
    async with db() as s:
        # Unnamed: the default-backend lookup already skips the disabled backend, so none is left.
        err = await queue_training(s, project_id=project_id, name="r", task=TASK, dataset_version_id=version_id)
        assert isinstance(err, QueueError) and "No trainer backend is available" in err.message
        # Named explicitly, it would bypass that lookup, so queue_training checks the switch itself.
        err = await queue_training(
            s, project_id=project_id, name="r", task=TASK, dataset_version_id=version_id, backend_id="ludwig"
        )
        assert isinstance(err, QueueError) and "disabled" in err.message
    assert await count(db, TrainingRun) == 0
