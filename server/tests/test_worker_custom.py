"""A bring-your-own model on a worker: fetched here with the server's own safe code, and the config re-pointed at it."""

import uuid
from pathlib import Path

import pytest

from omnitrain.services import custom_models
from omnitrain.settings import get_settings
from omnitrain.worker import custom

SERVER = "/srv/omnitrain/custom_models/m1/c0ffee"


def info(**fields):
    return {
        "id": str(uuid.uuid4()),
        "kind": "hf_video",
        "sourceKind": "upload",
        "sourceRef": None,
        "revision": None,
        "sha256": "a" * 64,
        "spec": {},
        "serverPath": SERVER,
        **fields,
    }


# -- relocate ------------------------------------------------------------------------------------


def test_relocate_swaps_the_server_prefix_at_any_depth_and_leaves_everything_else():
    config = {
        "custom": {"kind": "hf_video", "path": SERVER},
        "files": [f"{SERVER}/config.json", "/elsewhere/config.json", 3, None],
        "train": {"epochs": 2, "name": "run"},
    }

    assert custom.relocate(config, SERVER, "/w/models/m1") == {
        "custom": {"kind": "hf_video", "path": "/w/models/m1"},
        "files": ["/w/models/m1/config.json", "/elsewhere/config.json", 3, None],
        "train": {"epochs": 2, "name": "run"},
    }
    assert config["custom"]["path"] == SERVER  # the original is not modified


def test_relocate_only_rewrites_a_string_that_starts_with_the_path():
    assert custom.relocate({"note": f"see {SERVER}"}, SERVER, "/w") == {"note": f"see {SERVER}"}


# -- prepare -------------------------------------------------------------------------------------


@pytest.fixture
def worker_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path / "worker")
    return tmp_path / "worker"


def test_prepare_fetches_the_model_here_and_points_the_config_at_it(monkeypatch, worker_dir):
    fetched = []

    def fake_upload(row, dest: Path):
        fetched.append(row)
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(custom_models, "_fetch_upload", fake_upload)
    model = info()

    config = custom.prepare(model, {"custom": {"kind": "hf_video", "path": SERVER}, "epochs": 2})

    local = Path(config["custom"]["path"])
    assert worker_dir in local.parents and local.parts[-3] == "custom_models" and local.parts[-2] == model["id"]
    assert (local / "config.json").is_file()
    assert config["epochs"] == 2
    assert len(fetched) == 1 and fetched[0].sha256 == "a" * 64


def test_prepare_does_not_fetch_again_for_a_second_run_on_the_same_model(monkeypatch, worker_dir):
    calls = []
    monkeypatch.setattr(custom_models, "_fetch_upload", lambda row, dest: calls.append(row.id))
    model = info()

    first = custom.prepare(model, {"path": SERVER})
    second = custom.prepare(model, {"path": SERVER})

    assert first == second and len(calls) == 1


def test_a_hub_model_is_fetched_anonymously_at_its_pinned_revision(monkeypatch, worker_dir):
    seen = {}

    def fake_hub(row, dest: Path, token=None):
        seen.update(repo=row.source_ref, revision=row.revision, token=token)
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(custom_models, "_fetch_hub", fake_hub)

    custom.prepare(info(sourceKind="hub", sourceRef="MCG-NJU/videomae-base", revision="c0ffee", sha256=None), {})

    assert seen == {"repo": "MCG-NJU/videomae-base", "revision": "c0ffee", "token": None}


def test_a_token_in_the_workers_environment_is_never_sent_to_the_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_machine_wide_secret")
    assert custom_models._hub_token(None) is False
