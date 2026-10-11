"""The Hugging Face integration: saving a token (encrypted, never returned), and using it, and only the
owner's, to import and fetch models their account can read but the public cannot. Hugging Face itself is
faked; nothing here touches the network."""

import uuid

import httpx
import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from omnitrain.db.models import CustomModel, UserIntegration
from omnitrain.jobs import validate_model
from omnitrain.services import custom_models as cm
from omnitrain.services import integrations, run_models, secrets
from omnitrain.settings import get_settings

TOKEN = "hf_abcdefghijklmnopqrstuvwxyz0123"


def response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/x"))


@pytest.fixture
def hf_accepts(monkeypatch):
    """Hugging Face accepts every token and says it belongs to `alice`."""
    monkeypatch.setattr(integrations, "_check_token", lambda token: "alice")


# -- Saving a token ------------------------------------------------------------------------------


async def test_a_new_user_is_not_connected(new_user):
    user = await new_user()
    assert (await user.get("/api/integrations/huggingface")).json() == {
        "connected": False,
        "accountName": None,
        "tokenHint": None,
        "connectedAt": None,
    }


async def test_connecting_saves_the_token_encrypted_and_never_sends_it_back(new_user, db, hf_accepts):
    user = await new_user()
    r = await user.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] is True and body["accountName"] == "alice" and body["tokenHint"] == TOKEN[-4:]
    assert TOKEN not in r.text and TOKEN not in (await user.get("/api/integrations/huggingface")).text

    async with db() as s:
        row = (await s.execute(sa.select(UserIntegration))).scalar_one()
    assert row.token_encrypted != TOKEN and TOKEN not in row.token_encrypted  # encrypted at rest
    assert secrets.decrypt(row.token_encrypted) == TOKEN


async def test_connecting_again_replaces_the_token(new_user, db, hf_accepts):
    user = await new_user()
    await user.put("/api/integrations/huggingface", json={"token": TOKEN})
    other = "hf_" + "z" * 30
    assert (await user.put("/api/integrations/huggingface", json={"token": other})).json()["tokenHint"] == "zzzz"
    async with db() as s:
        rows = (await s.execute(sa.select(UserIntegration))).scalars().all()
        assert len(rows) == 1 and await integrations.huggingface_token(s, uuid.UUID(user.user_id)) == other


async def test_a_token_hugging_face_refuses_is_not_saved(new_user, db, monkeypatch):
    def refuse(token):
        raise HTTPException(400, "Hugging Face did not accept this token.")

    monkeypatch.setattr(integrations, "_check_token", refuse)
    user = await new_user()
    r = await user.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert r.status_code == 400 and "did not accept" in r.json()["error"]["message"]
    async with db() as s:
        assert (await s.execute(sa.select(sa.func.count()).select_from(UserIntegration))).scalar_one() == 0


@pytest.mark.parametrize("token", ["not-a-token", "hf_ has space", "hf_x", ""])
async def test_something_that_is_not_a_token_is_refused_before_hugging_face_is_asked(new_user, monkeypatch, token):
    monkeypatch.setattr(integrations, "_check_token", lambda t: pytest.fail("must not be asked"))
    user = await new_user()
    assert (await user.put("/api/integrations/huggingface", json={"token": token})).status_code == 422


async def test_disconnecting_forgets_the_token(new_user, db, hf_accepts):
    user = await new_user()
    await user.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert (await user.delete("/api/integrations/huggingface")).status_code == 204
    assert (await user.get("/api/integrations/huggingface")).json()["connected"] is False
    assert (await user.delete("/api/integrations/huggingface")).status_code == 204  # idempotent
    async with db() as s:
        assert await integrations.huggingface_token(s, uuid.UUID(user.user_id)) is None


async def test_one_users_token_is_invisible_to_another(new_user, hf_accepts):
    a, b = await new_user(), await new_user()
    await a.put("/api/integrations/huggingface", json={"token": TOKEN})
    assert (await b.get("/api/integrations/huggingface")).json()["connected"] is False


async def test_the_integration_needs_a_login(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as anon:
        assert (await anon.get("/api/integrations/huggingface")).status_code == 401
        assert (await anon.put("/api/integrations/huggingface", json={"token": TOKEN})).status_code == 401


# -- The encryption itself -----------------------------------------------------------------------


def test_secrets_round_trip_and_a_changed_key_reads_as_gone(monkeypatch):
    sealed = secrets.encrypt(TOKEN)
    assert sealed != TOKEN and secrets.decrypt(sealed) == TOKEN
    assert secrets.encrypt(TOKEN) != sealed  # a fresh nonce each time

    monkeypatch.setattr(get_settings(), "jwt_secret", "a-different-secret")
    assert secrets.decrypt(sealed) is None  # not an error: the user just reconnects
    assert secrets.decrypt("garbage") is None


def test_an_explicit_secrets_key_wins_over_the_jwt_derived_one(monkeypatch):
    from cryptography.fernet import Fernet

    sealed = secrets.encrypt(TOKEN)
    monkeypatch.setattr(get_settings(), "secrets_key", Fernet.generate_key().decode())
    assert secrets.decrypt(sealed) is None
    assert secrets.decrypt(secrets.encrypt(TOKEN)) == TOKEN


# -- Using the token to look a model up ----------------------------------------------------------


class FakeInfo:
    def __init__(self, *, private=False, gated=False):
        self.private, self.gated, self.sha = private, gated, "c0ffee"
        self.siblings = [
            type("S", (), {"rfilename": "config.json", "size": 10})(),
            type("S", (), {"rfilename": "model.safetensors", "size": 90})(),
        ]


def fake_model_info(monkeypatch, *, info=None, raises=None):
    """Replace HfApi.model_info, recording the `token` it was called with."""
    seen: list = []

    def model_info(self, repo_id, **kw):
        seen.append(kw.get("token"))
        if raises is not None:
            raise raises
        return info

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", model_info)
    return seen


def test_without_a_token_the_hub_is_asked_with_none_not_the_servers_own(monkeypatch):
    seen = fake_model_info(monkeypatch, info=FakeInfo())
    cm.hub_pin("org/public")
    assert seen == [False]  # False means "no token"; None would let huggingface_hub use one from the environment


def test_a_private_repository_needs_a_token_and_says_so(monkeypatch):
    fake_model_info(monkeypatch, info=FakeInfo(private=True))
    with pytest.raises(cm.CustomModelFileError, match="Connect your Hugging Face account in Settings"):
        cm.hub_pin("org/secret")


def test_a_private_repository_is_pinned_with_the_owners_token(monkeypatch):
    seen = fake_model_info(monkeypatch, info=FakeInfo(private=True))
    pin = cm.hub_pin("org/secret", TOKEN)
    assert seen == [TOKEN] and pin.private is True and pin.revision == "c0ffee"
    assert pin.size_bytes == 100 and pin.has_safetensors and pin.has_config


def test_a_public_repository_is_not_marked_private(monkeypatch):
    fake_model_info(monkeypatch, info=FakeInfo())
    assert cm.hub_pin("org/public", TOKEN).private is False


def test_a_gated_repository_the_account_can_read_is_usable_and_marked(monkeypatch):
    fake_model_info(monkeypatch, info=FakeInfo(gated="manual"))
    assert cm.hub_pin("org/gated", TOKEN).private is True


@pytest.mark.parametrize(
    ("token", "error", "message"),
    [
        (None, RepositoryNotFoundError("x", response=response(404)), "If it is private, connect"),
        (TOKEN, RepositoryNotFoundError("x", response=response(404)), "account cannot read it"),
        (None, GatedRepoError("x", response=response(403)), "Connect your Hugging Face account"),
        (TOKEN, GatedRepoError("x", response=response(403)), "has not been granted access"),
        (None, HfHubHTTPError("x", response=response(401)), "needs a Hugging Face account"),
        (TOKEN, HfHubHTTPError("x", response=response(401)), "refused your saved token"),
    ],
)
def test_a_refusal_from_the_hub_tells_the_user_what_to_do(monkeypatch, token, error, message):
    fake_model_info(monkeypatch, raises=error)
    with pytest.raises(cm.CustomModelFileError, match=message):
        cm.hub_pin("org/secret", token)


def test_an_unrelated_hub_failure_is_not_disguised_as_an_access_problem(monkeypatch):
    fake_model_info(monkeypatch, raises=HfHubHTTPError("boom", response=response(500)))
    with pytest.raises(HfHubHTTPError):  # propagates, so the job retries instead of failing the model
        cm.hub_pin("org/x", TOKEN)


async def test_inspecting_uses_the_callers_own_token_only(new_user, monkeypatch, hf_accepts):
    seen = fake_model_info(monkeypatch, info=FakeInfo(private=True))
    monkeypatch.setattr(cm, "fetch_hub_config", lambda repo_id, revision, token=None: None)
    connected, stranger = await new_user(), await new_user()
    await connected.put("/api/integrations/huggingface", json={"token": TOKEN})

    out = (await connected.post("/api/models/inspect", json={"sourceRef": "org/secret"})).json()
    assert out["errors"] == [] and out["private"] is True and seen == [TOKEN]

    out = (await stranger.post("/api/models/inspect", json={"sourceRef": "org/secret"})).json()
    assert out["private"] is False and any("private" in e for e in out["errors"])
    assert seen == [TOKEN, False]  # the stranger's lookup went out with no token at all, not the other user's


async def test_a_model_made_for_everyone_is_looked_up_without_any_token(new_user, monkeypatch, hf_accepts):
    seen = fake_model_info(monkeypatch, info=FakeInfo(private=True))
    admin = await new_user()
    await admin.put("/api/integrations/huggingface", json={"token": TOKEN})
    out = (await admin.post("/api/models/inspect", json={"sourceRef": "org/secret", "shared": True})).json()
    assert seen == [False] and any("private" in e for e in out["errors"])


# -- Using the token to fetch a model ------------------------------------------------------------


def hub_row(**kw) -> CustomModel:
    fields = {
        "id": uuid.uuid4(),
        "backend": "ludwig",
        "kind": "hf_transformer",
        "name": "x",
        "source_kind": "hub",
        "source_ref": "org/secret",
        "spec": {"materialize": True},
        **kw,
    }
    return CustomModel(**fields)


def test_validating_a_private_model_fetches_it_with_the_token_and_records_that_it_is_private(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)
    calls: dict = {}

    def pin(repo_id, token=None):
        calls["pin"] = token
        return cm.HubPin(revision="c0ffee", size_bytes=10, has_safetensors=True, has_config=True, private=True)

    def fetch(row, dest, token=None):
        calls["fetch"] = token
        (dest / "config.json").write_text('{"model_type": "bert", "architectures": ["BertModel"]}')
        (dest / "model.safetensors").write_bytes(b"\0" * 8)

    monkeypatch.setattr(cm, "hub_pin", pin)
    monkeypatch.setattr(cm, "_fetch_hub", fetch)
    result = validate_model.validate_sync(hub_row(), TOKEN)
    assert calls == {"pin": TOKEN, "fetch": TOKEN} and result.private is True


async def test_the_validation_job_loads_the_owners_token_and_keeps_the_private_flag(
    new_user, db, monkeypatch, hf_accepts
):
    owner = await new_user()
    await owner.put("/api/integrations/huggingface", json={"token": TOKEN})
    async with db() as s:
        row = hub_row(owner_user_id=uuid.UUID(owner.user_id), status="validating")
        s.add(row)
        await s.commit()
        model_id = row.id

    seen: list = []

    def validate(row, token=None):
        seen.append(token)
        return validate_model.Validated(revision="c0ffee", size_bytes=10, private=True)

    monkeypatch.setattr(validate_model, "validate_sync", validate)
    await validate_model.run_validate(model_id)

    assert seen == [TOKEN]
    async with db() as s:
        saved = await s.get(CustomModel, model_id)
    assert saved.status == "ready" and saved.spec == {"materialize": True, "private": True}
    models = (await owner.get("/api/models")).json()["models"]
    assert [m["private"] for m in models] == [True]


async def test_a_global_model_never_gets_anyones_token(db, monkeypatch):
    async with db() as s:
        row = hub_row(owner_user_id=None, status="validating")
        s.add(row)
        await s.commit()
        model_id = row.id
    seen: list = []

    def validate(row, token=None):
        seen.append(token)
        return validate_model.Validated(revision="c0ffee", size_bytes=10)

    monkeypatch.setattr(validate_model, "validate_sync", validate)
    await validate_model.run_validate(model_id)
    assert seen == [None]


async def test_a_run_refetches_a_private_model_with_its_owners_token(new_user, db, monkeypatch, hf_accepts):
    owner, stranger = await new_user(), await new_user()
    await owner.put("/api/integrations/huggingface", json={"token": TOKEN})
    await stranger.put("/api/integrations/huggingface", json={"token": "hf_" + "s" * 30})
    async with db() as s:
        row = hub_row(owner_user_id=uuid.UUID(owner.user_id), revision="c0ffee", status="ready")
        s.add(row)
        await s.flush()
        project_id, version_id = await _project_and_version(s, uuid.UUID(owner.user_id))
        run = _run(project_id, version_id, row.id)
        s.add(run)
        await s.commit()
        run_id = run.id

    seen: list = []
    monkeypatch.setattr(cm, "ensure_local", lambda row, token=None: seen.append(token))
    await run_models.ensure_run_custom_model(run_id)
    assert seen == [TOKEN]  # the owner's, not the other user's


async def test_a_revoked_token_fails_the_refetch_with_a_message_not_a_server_error(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "temp_dir", tmp_path)

    def snapshot_download(**kw):
        assert kw["token"] is False  # no saved token, and the server's own is never substituted
        raise HfHubHTTPError("nope", response=response(401))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    with pytest.raises(cm.CustomModelFileError, match="needs a Hugging Face account"):
        cm.ensure_local(hub_row(revision="c0ffee"), None)


# -- Helpers for the run test --------------------------------------------------------------------


async def _project_and_version(session, user_id):
    from omnitrain.db.models import Dataset, DatasetVersion, Project

    project = Project(user_id=user_id, name="p", task="text_classification")
    session.add(project)
    await session.flush()
    session.add(Dataset(project_id=project.id, modality="text"))
    await session.flush()
    version = DatasetVersion(dataset_id=project.id, status="draft")
    session.add(version)
    await session.flush()
    return project.id, version.id


def _run(project_id, version_id, custom_model_id):
    from omnitrain.db.models import TrainingRun

    return TrainingRun(
        project_id=project_id,
        dataset_version_id=version_id,
        name="r",
        status="succeeded",
        hyperparameters={},
        custom_model_id=custom_model_id,
    )
