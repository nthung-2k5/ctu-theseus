"""Bring-your-own models without a database: safe unpacking and the generic file checks.

What each backend adds on top (its own validation and how a kind compiles) is tested with that backend:
`test_pytorch_custom*.py`, `test_llm_backend.py`."""

import json
import stat
import zipfile
from pathlib import Path

import pytest

from omnitrain.plugins.backends.base import CustomModelRef
from omnitrain.services import custom_models as cm

MB = 2**20


def make_zip(path: Path, files: dict[str, bytes], *, symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3  # unix, so external_attr carries the file mode
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/etc/passwd")
    return path


def write_hf_model(
    root: Path,
    *,
    model_type: str | None = "bert",
    architectures: list[str] | None = None,
    weights: bool = True,
    extra_config: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config: dict = {"architectures": architectures or ["BertModel"], **(extra_config or {})}
    if model_type is not None:
        config["model_type"] = model_type
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    if weights:
        (root / "model.safetensors").write_bytes(b"\0" * 16)
    return root


def ref(kind: str, path: str = "/models/x", **kw) -> CustomModelRef:
    return CustomModelRef(id="custom:1", kind=kind, source_kind=kw.pop("source_kind", "hub"), local_path=path, **kw)


# -- Unpacking ---------------------------------------------------------------------------------


class TestExtractBundle:
    def test_a_single_top_level_folder_is_stripped(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"my-model/config.json": b"{}", "my-model/model.safetensors": b"x"})
        cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["config.json", "model.safetensors"]

    def test_files_at_the_top_level_are_left_where_they_are(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"config.json": b"{}", "sub/tokenizer.json": b"{}"})
        cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)
        assert (tmp_path / "out" / "config.json").is_file() and (tmp_path / "out" / "sub" / "tokenizer.json").is_file()

    def test_a_path_that_climbs_out_of_the_folder_is_refused(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"ok/config.json": b"{}", "ok/../../evil.txt": b"x"})
        with pytest.raises(cm.CustomModelFileError, match="outside"):
            cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)
        assert not (tmp_path / "evil.txt").exists()

    def test_an_absolute_path_is_refused(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"/etc/evil": b"x", "config.json": b"{}"})
        with pytest.raises(cm.CustomModelFileError, match="outside"):
            cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)

    def test_a_symbolic_link_is_refused(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"config.json": b"{}"}, symlink="link")
        with pytest.raises(cm.CustomModelFileError, match="symbolic link"):
            cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)

    def test_the_limit_applies_to_the_unzipped_size(self, tmp_path):
        z = make_zip(tmp_path / "a.zip", {"big.bin": b"0" * (2 * MB)})  # compresses to almost nothing
        with pytest.raises(cm.CustomModelFileError, match="larger than"):
            cm.extract_bundle(z, tmp_path / "out", max_bytes=MB)

    def test_not_a_zip_and_an_empty_zip_are_refused(self, tmp_path):
        (tmp_path / "junk.zip").write_bytes(b"not a zip")
        with pytest.raises(cm.CustomModelFileError, match="not a valid zip"):
            cm.extract_bundle(tmp_path / "junk.zip", tmp_path / "o1", max_bytes=MB)
        with pytest.raises(cm.CustomModelFileError, match="empty"):
            cm.extract_bundle(make_zip(tmp_path / "e.zip", {}), tmp_path / "o2", max_bytes=MB)


class TestCheckTree:
    @pytest.mark.parametrize(
        "name", ["pytorch_model.bin", "model.pt", "weights.pth", "m.ckpt", "x.pkl", "run.py", "x.so"]
    )
    def test_pickles_and_code_are_refused(self, tmp_path, name):
        write_hf_model(tmp_path)
        (tmp_path / name).write_bytes(b"x")
        with pytest.raises(cm.CustomModelFileError, match="not accepted"):
            cm.check_tree(tmp_path, max_bytes=MB)

    def test_a_clean_huggingface_folder_passes_and_reports_its_size(self, tmp_path):
        write_hf_model(tmp_path)
        (tmp_path / cm._COMPLETE_MARKER).touch()  # bookkeeping, not part of the model
        assert (
            cm.check_tree(tmp_path, max_bytes=MB)
            == len("{}") + len(json.dumps({"architectures": ["BertModel"], "model_type": "bert"})) + 16
        )

    def test_the_size_limit_and_an_empty_folder(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"0" * (2 * MB))
        with pytest.raises(cm.CustomModelFileError, match="over the"):
            cm.check_tree(tmp_path, max_bytes=MB)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(cm.CustomModelFileError, match="no files"):
            cm.check_tree(empty, max_bytes=MB)

    def test_a_symlinked_file_is_refused(self, tmp_path):
        write_hf_model(tmp_path)
        try:
            (tmp_path / "link.json").symlink_to(tmp_path / "config.json")
        except (OSError, NotImplementedError):
            pytest.skip("cannot create symlinks here")
        with pytest.raises(cm.CustomModelFileError, match="symbolic link"):
            cm.check_tree(tmp_path, max_bytes=MB)


def test_sha256_is_the_content_hash(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    assert cm.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
