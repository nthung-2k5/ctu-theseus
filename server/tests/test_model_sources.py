"""Where a bring-your-own model comes from is a plugin: `omnitrain/plugins/model_sources/`."""

import uuid

import pytest
import sqlalchemy as sa

from omnitrain.db.models import CustomModel
from omnitrain.plugins.model_sources import (
    ModelSource,
    SourceRefError,
    Validated,
    find_source,
    get_source,
    list_sources,
)
from omnitrain.plugins.model_sources.base import _registry
from omnitrain.services import custom_models
from omnitrain.services.custom_models import CustomModelFileError


@pytest.fixture
def sources():
    _registry.ensure_loaded()
    before = set(_registry._items)
    yield _registry
    for plugin_id in set(_registry._items) - before:
        _registry.unregister(plugin_id)


def _row(source_kind: str, **kw) -> CustomModel:
    return CustomModel(id=uuid.uuid4(), kind="k", source_kind=source_kind, spec={}, **kw)


def test_hub_and_upload_are_the_shipped_sources():
    assert [s.id for s in list_sources()] == ["hub", "upload"]
    assert get_source("hub").needs_ref and not get_source("hub").needs_upload
    assert get_source("upload").needs_upload and not get_source("upload").needs_ref
    assert find_source("ftp") is None
    with pytest.raises(KeyError):
        get_source("ftp")


class TestReferences:
    def test_a_hub_reference_is_a_repository_id_and_is_trimmed(self):
        hub = get_source("hub")
        assert hub.normalize_ref("  org/name ") == "org/name"
        assert hub.normalize_ref("bert-base-uncased") == "bert-base-uncased"
        for bad in (None, "", "   ", "a/b/c", "has space", "/leading"):
            with pytest.raises(SourceRefError, match="repository id"):
                hub.normalize_ref(bad)

    def test_an_upload_takes_no_reference(self):
        upload = get_source("upload")
        assert upload.normalize_ref(None) is None
        assert upload.normalize_ref("  ") is None
        with pytest.raises(SourceRefError, match="no source reference"):
            upload.normalize_ref("org/name")


class TestFetching:
    def test_ensure_local_asks_the_rows_source_for_the_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(custom_models, "local_dir_for", lambda row: tmp_path / row.source_kind)
        calls = []
        monkeypatch.setattr(custom_models, "_fetch_hub", lambda row, dest, token: calls.append(("hub", token)))
        monkeypatch.setattr(custom_models, "_fetch_upload", lambda row, dest: calls.append(("upload", None)))

        custom_models.ensure_local(_row("hub"), token="t")
        custom_models.ensure_local(_row("upload"))
        assert calls == [("hub", "t"), ("upload", None)]

    def test_a_source_that_is_not_installed_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(custom_models, "local_dir_for", lambda row: tmp_path / "m")
        with pytest.raises(CustomModelFileError, match="'ftp', which is not an installed source"):
            custom_models.ensure_local(_row("ftp"))

    def test_a_new_source_is_used_with_no_other_edit(self, sources, monkeypatch, tmp_path):
        monkeypatch.setattr(custom_models, "local_dir_for", lambda row: tmp_path / "m")

        class Url(ModelSource):
            id = "url"
            label = "A URL"
            needs_ref = True

            @classmethod
            def normalize_ref(cls, ref):
                if not (ref or "").startswith("https://"):
                    raise SourceRefError("Enter an https:// URL")
                return ref

            @classmethod
            def fetch(cls, row, dest, token):
                (dest / "config.json").write_text(row.source_ref)

        assert get_source("url") is Url and "url" in [s.id for s in list_sources()]
        path = custom_models.ensure_local(_row("url", source_ref="https://example.com/m"))
        assert (path / "config.json").read_text() == "https://example.com/m"

    def test_a_source_without_a_label_or_with_a_taken_id_is_refused(self, sources):
        with pytest.raises(TypeError, match="needs a label"):

            class Unlabelled(ModelSource):
                id = "unlabelled"

        with pytest.raises(ValueError, match="Duplicate model source id"):

            class Again(ModelSource):
                id = "hub"
                label = "Another hub"


class TestValidation:
    def test_an_upload_is_pinned_by_the_hash_of_its_bundle(self, monkeypatch, tmp_path):
        imported = custom_models.ImportedUpload(sha256="ab" * 32, size_bytes=3, path=tmp_path)
        monkeypatch.setattr(custom_models, "import_upload", lambda row: imported)
        row = _row("upload")
        result, path = get_source("upload").validate(row, None, 10**9)
        assert (result.sha256, result.revision, path) == ("ab" * 32, None, tmp_path)
        assert row.sha256 == "ab" * 32

    def test_a_hub_model_is_pinned_to_a_commit_and_reports_whether_it_is_private(self, monkeypatch, tmp_path):
        pin = custom_models.HubPin(
            revision="c0ffee", size_bytes=10, has_safetensors=True, has_config=True, private=True
        )
        seen = {}
        monkeypatch.setattr(custom_models, "hub_pin", lambda ref, token: seen.update(ref=ref, token=token) or pin)
        monkeypatch.setattr(custom_models, "ensure_local", lambda row, token: tmp_path)
        row = _row("hub", source_ref="org/name")
        result, path = get_source("hub").validate(row, "secret", 1000)
        assert result == Validated(revision="c0ffee", private=True) and path == tmp_path
        assert row.revision == "c0ffee" and seen == {"ref": "org/name", "token": "secret"}

    def test_an_oversized_hub_model_is_refused_before_anything_is_downloaded(self, monkeypatch):
        pin = custom_models.HubPin(revision="c0ffee", size_bytes=5 * 2**20, has_safetensors=True, has_config=True)
        monkeypatch.setattr(custom_models, "hub_pin", lambda ref, token: pin)
        monkeypatch.setattr(custom_models, "ensure_local", lambda *a: pytest.fail("must not download"))
        with pytest.raises(CustomModelFileError, match="over the 1 MiB limit"):
            get_source("hub").validate(_row("hub", source_ref="org/name"), None, 2**20)


async def test_only_the_hub_source_uses_a_token_and_a_shared_model_has_none(monkeypatch):
    from omnitrain.services import integrations

    async def fake(owner):
        return None if owner is None else f"token-of-{owner}"

    monkeypatch.setattr(integrations, "huggingface_token_for", fake)
    owner = uuid.uuid4()
    assert await get_source("hub").token_for(owner) == f"token-of-{owner}"
    assert await get_source("hub").token_for(None) is None
    assert await get_source("upload").token_for(owner) is None
    assert get_source("hub").shared_failure_hint and not get_source("upload").shared_failure_hint


def test_a_worker_may_fetch_an_upload_and_a_public_hub_model_but_not_a_private_one():
    """The SQL each source contributes to the queue's `_worker_can_take_model`."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.orm import aliased

    model = aliased(CustomModel)
    sql = {
        s.id: str(
            s.worker_fetchable(model).compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        for s in list_sources()
    }
    assert sql["upload"] == "true"
    assert "private" in sql["hub"] and "coalesce" in sql["hub"].lower()
    # A source that says nothing is never offered to a worker.
    assert str(ModelSource.worker_fetchable(model)) == str(sa.false())
