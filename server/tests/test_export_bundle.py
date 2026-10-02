"""Export bundles: per-format file layout, preprocessing decompilation, zip, and assembly end to end."""

import hashlib
import io
import json
import uuid
import zipfile

import pytest
import sqlalchemy as sa

from theseus.backends.ludwig.manifest import build_preprocessing_manifest as build_manifest
from theseus.db.models import LabelClass, ModelExport, TrainingRun
from theseus.export import bundle as B
from theseus.export.formats.base import ExportFormat, _registry
from theseus.export.formats.csharp_devkit import CSharpDevkit
from theseus.export.formats.pwa_app import PwaApp
from theseus.export.formats.python_devkit import PythonDevkit
from theseus.export.metadata import extract_preprocessing
from theseus.export.readme import ReadmeVars, app_readme, devkit_readme, model_readme
from theseus.export.registry import get_export_format, list_export_formats
from theseus.jobs import export as export_job
from theseus.services import storage

CLASSES = ["dog", "cat"]  # Ludwig order: deliberately NOT alphabetical
PRE = {
    "schemaVersion": 1,
    "runId": "r1",
    "inputs": [{"name": "image_path", "type": "image", "column": "image_path"}],
    "outputs": [{"name": "class", "type": "category", "column": "class", "classes": CLASSES}],
}
GOLDEN = B.GoldenSample(b'{"schemaVersion": 1}', "input.png", b"png-bytes")


def files(fmt="python_devkit", golden=GOLDEN, pre=PRE, name="Cats vs Dogs"):
    out = B.assemble_files(
        run_name=name, task_label="Image Classification", format_id=fmt,
        model_bytes=b"MODEL", preprocessing=pre, golden=golden,
    )  # fmt: skip
    return {f.path: f for f in out}


# -- Pure helpers ----------------------------------------------------------------------------


def test_render_substitutes_known_vars_and_blanks_unknown_ones():
    assert B.render("Hi {{RUN_NAME}} / {{MISSING}} / {{ NOT }}", {"RUN_NAME": "x"}) == "Hi x /  / {{ NOT }}"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("My Model 2", "my_model_2"), ("  --Cats & Dogs!!  ", "cats_dogs"), ("123 run", "app_123_run"),
     ("!!!", "theseus_app"), ("", "theseus_app"), ("Über", "ber")],
)  # fmt: skip
def test_dart_package_names_are_valid_lowercase_snake_case(name, expected):
    assert B.dart_package_name(name) == expected


def test_sample_file_resolution():
    calls = []

    def download(bucket, key):
        calls.append((bucket, key))
        return b"bytes"

    assert B.resolve_sample_file("s3://theseus-datasets/pool/p/ab/abc.PNG", download) == ("input.PNG", b"bytes")
    assert calls == [("theseus-datasets", "pool/p/ab/abc.PNG")]
    assert B.resolve_sample_file("s3://bucket/noext", download) == ("input", b"bytes")
    assert B.resolve_sample_file("hello world", download) == ("input.txt", b"hello world")
    assert B.resolve_sample_file({"age": 30}, download) == ("input.json", json.dumps({"age": 30}, indent=2).encode())
    assert B.resolve_sample_file(None, download) is None
    assert B.resolve_sample_file("s3://bucketonly", download) is None

    def broken(bucket, key):
        raise OSError("gone")

    assert B.resolve_sample_file("s3://b/k.png", broken) is None  # a missing sample just disables verify


def test_golden_expected_json_is_rewritten_to_bundle_local_paths():
    raw = {"inputColumn": "x", "inputValue": "s3://b/k.png", "outputColumn": "class", "outputType": "category",
           "predictions": {"cat": 0.9}}  # fmt: skip
    out = json.loads(B.to_bundle_local(raw, "input.png"))
    assert out == {"schemaVersion": 1, "sampleFile": "sample/input.png", "outputColumn": "class",
                   "outputType": "category", "predictions": {"cat": 0.9}}  # fmt: skip
    assert "s3://" not in json.dumps(out)  # no internal bucket path leaks into a user-facing bundle


# -- Layout per format -----------------------------------------------------------------------


def test_model_formats_are_just_the_artifact_metadata_labels_and_readme():
    assert set(files("onnx")) == {"model.onnx", "preprocessing.json", "labels.txt", "README.md"}
    assert set(files("torch_export")) == {"model.pt2", "preprocessing.json", "labels.txt", "README.md"}


def test_labels_txt_is_written_in_ludwig_index_order_and_only_for_classification():
    assert files("onnx")["labels.txt"].data == b"dog\ncat"
    regression = {**PRE, "outputs": [{"name": "target", "type": "number", "column": "target"}]}
    assert "labels.txt" not in files("onnx", pre=regression)


@pytest.mark.parametrize(
    ("fmt", "client_files", "verify"),
    [
        ("python_devkit", {"theseus_client.py", "example.py"}, "verify.py"),
        ("typescript_devkit", {"client.ts", "example.ts"}, "verify.ts"),
        ("csharp_devkit", {"TheseusClient.cs", "Program.cs"}, None),
        ("java_devkit", {"TheseusClient.java", "Main.java"}, None),
    ],
)
def test_devkits_ship_source_only_with_verify_where_supported(fmt, client_files, verify):
    with_golden = set(files(fmt))
    base = {"model.onnx", "preprocessing.json", "labels.txt", "README.md", "expected.json", "sample/input.png"}
    assert with_golden == base | client_files | ({verify} if verify else set())
    without = set(files(fmt, golden=None))
    assert without == {"model.onnx", "preprocessing.json", "labels.txt", "README.md"} | client_files
    # devkits are deliberately source only: never a build or project file
    assert not any(
        p.endswith((".csproj", ".gradle", ".kts", "package.json", "pyproject.toml", "Dockerfile")) for p in with_golden
    )


def test_pwa_places_everything_at_the_root_and_renders_its_templates():
    out = files("pwa_app", golden=GOLDEN)
    assert set(out) == {
        "index.html", "app.js", "sw.js", "manifest.webmanifest", "icon.svg", "style.css",
        "model.onnx", "preprocessing.json", "labels.txt", "expected.json", "sample/input.png", "README.md",
    }  # fmt: skip
    assert b"Cats vs Dogs" in out["index.html"].data and b"{{" not in out["index.html"].data
    assert b"Cats vs Dogs" in out["manifest.webmanifest"].data and b"{{" not in out["manifest.webmanifest"].data
    json.loads(out["manifest.webmanifest"].data)  # still valid JSON after substitution


def test_flutter_puts_model_files_under_assets_and_uses_a_valid_package_name():
    out = files("flutter_app", name="Cats vs Dogs")
    assert {"assets/model.onnx", "assets/preprocessing.json", "assets/labels.txt", "assets/expected.json",
            "assets/sample/input.png", "pubspec.yaml", "lib/main.dart", "lib/theseus_client.dart",
            "README.md"} == set(out)  # fmt: skip
    assert "model.onnx" not in out  # only reachable if declared as a pubspec asset
    pubspec = out["pubspec.yaml"].data.decode()
    assert "name: cats_vs_dogs" in pubspec and "{{" not in pubspec
    assert "{{" not in out["lib/main.dart"].data.decode()


def test_the_model_artifact_is_stored_uncompressed_and_everything_else_is_deflated():
    out = files("python_devkit")
    assert out["model.onnx"].compress is False
    assert all(f.compress for p, f in out.items() if p != "model.onnx")


def test_a_model_format_never_embeds_a_golden_sample():
    assert "expected.json" not in files("onnx", golden=GOLDEN)


def test_an_unknown_format_is_a_clear_error():
    with pytest.raises(KeyError, match="Unknown export format 'nope'"):
        files("nope")


def test_every_shipped_template_exists_and_is_valid_utf8():
    for path in sorted(p for p in B.TEMPLATES.rglob("*") if p.is_file()):
        assert path.read_text(encoding="utf-8")  # non-empty, decodable
    assert len(list(B.TEMPLATES.rglob("*.py"))) == 3 and (B.TEMPLATES / "flutter/lib/theseus_client.dart").exists()


def test_readme_reflects_the_format_and_whether_a_verify_path_exists():
    def v(verify):
        return ReadmeVars("R", "T", "Some format", "model.onnx", verify)

    dev = devkit_readme(v(True), PythonDevkit.readme_info)
    assert "pip install onnxruntime" in dev and "python verify.py" in dev and "`expected.json`" in dev
    assert "python verify.py" not in devkit_readme(v(False), PythonDevkit.readme_info)
    app = app_readme(v(True), PwaApp.readme_info)
    assert "Progressive Web App" in app and "self-check" in app
    assert "no client code" in model_readme(v(False)) and "`model.onnx`" in model_readme(v(False))
    assert "dotnet add package" in devkit_readme(v(False), CSharpDevkit.readme_info)


# -- Plugin discovery ------------------------------------------------------------------------


def test_the_shipped_formats_are_discovered_grouped_and_ordered():
    listed = [(f.group, f.id) for f in list_export_formats()]
    assert listed == [
        ("Model", "onnx"), ("Model", "torch_export"),
        ("Devkit", "python_devkit"), ("Devkit", "typescript_devkit"), ("Devkit", "csharp_devkit"),
        ("Devkit", "java_devkit"),
        ("App", "pwa_app"), ("App", "flutter_app"),
    ]  # fmt: skip


def test_a_new_format_class_is_picked_up_with_no_other_change():
    class ExtraFormat(ExportFormat):
        id = "test_only_format"
        label = "Test only"
        group = "Experimental"

        @classmethod
        def assemble(cls, ctx):
            ctx.add("hello.txt", "hi")

    try:
        assert get_export_format("test_only_format") is ExtraFormat
        assert list_export_formats()[-1] is ExtraFormat  # unknown groups sort after the built-in ones
        out = B.assemble_files(
            run_name="r", task_label="t", format_id="test_only_format", model_bytes=b"m", preprocessing=PRE, golden=None
        )
        assert [f.path for f in out] == ["hello.txt"]
    finally:
        _registry.unregister("test_only_format")


def test_a_duplicate_format_id_fails_loudly_at_import():
    list_export_formats()  # make sure the shipped formats are loaded before defining a clash
    with pytest.raises(ValueError, match="Duplicate export format id 'onnx'"):

        class Clash(ExportFormat):
            id = "onnx"
            label = "Clash"

            @classmethod
            def assemble(cls, ctx):
                pass


def test_a_helper_base_without_an_id_is_not_registered():
    class Helper(ExportFormat):
        label = "Helper"

        @classmethod
        def assemble(cls, ctx):
            pass

    assert Helper not in list_export_formats()


# -- Zip -------------------------------------------------------------------------------------


def test_zip_stores_the_model_deflates_the_rest_and_is_byte_for_byte_reproducible():
    listing = list(files("python_devkit").values())
    first, second = B.make_zip(listing), B.make_zip(listing)
    assert first == second  # a fixed epoch: identical inputs give an identical bundle and checksum
    with zipfile.ZipFile(io.BytesIO(first)) as zf:
        by_name = {i.filename: i for i in zf.infolist()}
        assert by_name["model.onnx"].compress_type == zipfile.ZIP_STORED
        assert by_name["README.md"].compress_type == zipfile.ZIP_DEFLATED
        assert zf.read("model.onnx") == b"MODEL" and zf.read("labels.txt") == b"dog\ncat"
        assert zf.testzip() is None


# -- Preprocessing decompilation -------------------------------------------------------------


def test_manifest_takes_class_order_from_ludwig_metadata_and_passes_preprocessing_through():
    cfg = {
        "input_features": [{"name": "image_path", "type": "image", "column": "image_path"}],
        "output_features": [{"name": "class", "type": "category", "column": "class"}],
    }
    meta = {
        "image_path": {"preprocessing": {"height": 224, "width": 224, "standardize_image": "imagenet1k"}},
        "class": {"idx2str": ["dog", "cat"], "str2idx": {"dog": 0, "cat": 1}},
    }
    m = build_manifest("r1", cfg, meta)
    assert m["outputs"][0]["classes"] == ["dog", "cat"]  # Ludwig order, never alphabetical
    inp = m["inputs"][0]
    assert inp["ludwigPreprocessing"]["height"] == 224
    assert inp["imageNormalization"] == {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def test_unknown_image_presets_and_missing_metadata_are_not_guessed_at():
    cfg = {"input_features": [{"name": "i", "type": "image", "column": "i"}], "output_features": []}
    m = build_manifest("r", cfg, {"i": {"preprocessing": {"standardize_image": "some_new_preset"}}})
    assert "imageNormalization" not in m["inputs"][0]
    assert m["inputs"][0]["ludwigPreprocessing"]["standardize_image"] == "some_new_preset"  # kept raw
    bare = build_manifest("r", cfg, None)
    assert bare["inputs"][0] == {"name": "i", "type": "image", "column": "i"} and bare["outputs"] == []


def test_number_inputs_carry_their_fitted_normalization_stats_from_the_metadata_siblings():
    cfg = {
        "input_features": [
            {"name": "age", "type": "number", "column": "age"},
            {"name": "raw", "type": "number", "column": "raw"},
            {"name": "img", "type": "image", "column": "img"},
        ],
        "output_features": [{"name": "target", "type": "number", "column": "target"}],
    }
    meta = {
        "age": {"preprocessing": {"normalization": "zscore"}, "mean": 40.5, "std": 12.0},
        "raw": {"preprocessing": {"normalization": None}},
        "img": {"preprocessing": {"normalization": "zscore"}, "mean": 1},
    }
    m = build_manifest("r", cfg, meta)
    assert m["inputs"][0]["numberNormalization"] == {"type": "zscore", "mean": 40.5, "std": 12.0}
    assert "numberNormalization" not in m["inputs"][1]  # normalization disabled
    assert "numberNormalization" not in m["inputs"][2]  # only number inputs
    assert "classes" not in m["outputs"][0]  # regression has no class list


def test_a_run_with_no_compiled_config_cannot_be_exported():
    with pytest.raises(ValueError, match="no compiled Ludwig config"):
        build_manifest("r", {}, None)


async def test_extract_preprocessing_ignores_the_label_classes_table_entirely(db, make_run, monkeypatch):
    """The wrong-order trap: Postgres sorted the classes alphabetically, Ludwig did not."""
    rid = await make_run(status="succeeded")
    async with db() as s:
        run = (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == rid))).scalar_one()
        run.config = {
            "input_features": [{"name": "text", "type": "text", "column": "text"}],
            "output_features": [{"name": "class", "type": "category", "column": "class"}],
        }
        s.add_all(
            [LabelClass(dataset_id=run.project_id, name="cat"), LabelClass(dataset_id=run.project_id, name="dog")]
        )
        await s.commit()
    meta = {"class": {"idx2str": ["dog", "cat"]}}
    monkeypatch.setattr(storage, "list_keys", lambda b, p: [f"{rid}/results/results_run_0/training_set_metadata.json"])
    monkeypatch.setattr(storage, "download_bytes", lambda b, k: json.dumps(meta).encode())
    async with db() as s:
        manifest = await extract_preprocessing(s, rid)
    assert manifest["outputs"][0]["classes"] == ["dog", "cat"]  # not ["cat", "dog"]


# -- Assembly end to end (real DB, fake S3) --------------------------------------------------


@pytest.fixture
def s3(monkeypatch):
    store: dict[tuple[str, str], bytes] = {}
    monkeypatch.setattr(storage, "upload_bytes", lambda b, k, d, content_type=None: store.__setitem__((b, k), d))
    monkeypatch.setattr(storage, "download_bytes", lambda b, k: store[(b, k)])
    monkeypatch.setattr(storage, "file_exists", lambda b, k: (b, k) in store)
    monkeypatch.setattr(storage, "list_keys", lambda b, p: [k for (bb, k) in store if bb == b and k.startswith(p)])
    return store


async def prepared_export(db, make_export, s3, *, fmt="python_devkit", golden=True, config=True):
    export_id, run_id = await make_export(status="assembling", attempt=1, fmt=fmt)
    async with db() as s:
        run = (await s.execute(sa.select(TrainingRun).where(TrainingRun.id == run_id))).scalar_one()
        run.name = "Cats vs Dogs"
        if config:
            run.config = {
                "input_features": [{"name": "image_path", "type": "image", "column": "image_path"}],
                "output_features": [{"name": "class", "type": "category", "column": "class"}],
            }
        await s.commit()
    s3[("theseus-models", f"{run_id}/model.onnx")] = b"ONNX-BYTES"
    s3[("theseus-training", f"{run_id}/results/results_run_0/training_set_metadata.json")] = json.dumps(
        {"class": {"idx2str": CLASSES}, "image_path": {"preprocessing": {"height": 8}}}
    ).encode()
    if golden:
        s3[("theseus-datasets", "pool/p/ab/x.png")] = b"the-real-sample"
        s3[("theseus-models", f"{run_id}/expected.json")] = json.dumps(
            {"inputColumn": "image_path", "inputValue": "s3://theseus-datasets/pool/p/ab/x.png",
             "outputColumn": "class", "outputType": "category", "predictions": {"dog": 0.7, "cat": 0.3}}
        ).encode()  # fmt: skip
    return export_id, run_id


async def row(db, export_id) -> ModelExport:
    async with db() as s:
        return (await s.execute(sa.select(ModelExport).where(ModelExport.id == export_id))).scalar_one()


async def test_build_bundle_produces_a_verified_devkit_zip_and_marks_the_export_ready(db, make_export, s3):
    export_id, run_id = await prepared_export(db, make_export, s3)
    await B.build_bundle(export_id)

    e = await row(db, export_id)
    assert (e.status, e.bundle_key) == ("ready", f"{run_id}/bundles/{export_id}.zip") and e.ready_at is not None
    zipped = s3[("theseus-models", e.bundle_key)]
    assert (
        e.byte_size == len(zipped) and e.checksum == hashlib.sha256(zipped).hexdigest()
    )  # checksum matches what was uploaded

    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        assert set(zf.namelist()) == {
            "model.onnx", "preprocessing.json", "labels.txt", "expected.json", "sample/input.png",
            "theseus_client.py", "example.py", "verify.py", "README.md",
        }  # fmt: skip
        assert zf.read("model.onnx") == b"ONNX-BYTES" and zf.read("sample/input.png") == b"the-real-sample"
        assert zf.read("labels.txt") == b"dog\ncat"  # Ludwig index order, from the metadata file
        expected = json.loads(zf.read("expected.json"))
        assert expected["sampleFile"] == "sample/input.png" and expected["predictions"] == {"dog": 0.7, "cat": 0.3}
        assert json.loads(zf.read("preprocessing.json"))["inputs"][0]["ludwigPreprocessing"] == {"height": 8}
        assert "Cats vs Dogs" in zf.read("README.md").decode()


async def test_a_devkit_without_a_golden_sample_omits_verify_but_still_builds(db, make_export, s3):
    export_id, _ = await prepared_export(db, make_export, s3, golden=False)
    await B.build_bundle(export_id)
    e = await row(db, export_id)
    with zipfile.ZipFile(io.BytesIO(s3[("theseus-models", e.bundle_key)])) as zf:
        names = set(zf.namelist())
    assert e.status == "ready" and "verify.py" not in names and "expected.json" not in names


async def test_a_golden_sample_whose_input_file_vanished_degrades_to_no_verify_instead_of_failing(db, make_export, s3):
    export_id, _ = await prepared_export(db, make_export, s3)
    del s3[("theseus-datasets", "pool/p/ab/x.png")]  # the dataset item was deleted after the export ran
    await B.build_bundle(export_id)
    e = await row(db, export_id)
    with zipfile.ZipFile(io.BytesIO(s3[("theseus-models", e.bundle_key)])) as zf:
        assert e.status == "ready" and "verify.py" not in zf.namelist()


async def test_a_model_format_needs_no_golden_sample_or_client(db, make_export, s3):
    export_id, _ = await prepared_export(db, make_export, s3, fmt="onnx")
    await B.build_bundle(export_id)
    e = await row(db, export_id)
    with zipfile.ZipFile(io.BytesIO(s3[("theseus-models", e.bundle_key)])) as zf:
        assert set(zf.namelist()) == {"model.onnx", "preprocessing.json", "labels.txt", "README.md"}


async def test_build_bundle_records_a_failure_instead_of_raising(db, make_export, s3):
    export_id, run_id = await prepared_export(db, make_export, s3)
    del s3[("theseus-models", f"{run_id}/model.onnx")]  # the converted artifact is missing
    await B.build_bundle(export_id)  # must not raise
    e = await row(db, export_id)
    assert e.status == "failed" and e.failed_message and e.bundle_key is None

    no_config, _ = await prepared_export(db, make_export, s3, config=False)
    await B.build_bundle(no_config)
    assert "no compiled Ludwig config" in (await row(db, no_config)).failed_message


async def test_a_stale_assembly_result_never_overwrites_a_row_that_moved_on(db, make_export, s3):
    export_id, _ = await prepared_export(db, make_export, s3)
    async with db() as s:  # startup recovery re-queued this export while the zip was being built
        await s.execute(sa.update(ModelExport).where(ModelExport.id == export_id).values(status="pending"))
        await s.commit()
    await B.build_bundle(export_id)
    e = await row(db, export_id)
    assert e.status == "pending" and e.bundle_key is None


async def test_the_whole_export_job_converts_then_assembles_to_ready(db, make_export, s3, monkeypatch):
    """run_export end to end with the real build_bundle (only the GPU conversion is stubbed)."""
    export_id, run_id = await prepared_export(db, make_export, s3)
    del s3[("theseus-models", f"{run_id}/model.onnx")]
    async with db() as sess:
        await sess.execute(sa.update(ModelExport).where(ModelExport.id == export_id).values(status="converting"))
        await sess.commit()

    def fake_convert(run, backend, artifact_id, dataset_key, export):
        artifact = backend.artifacts[artifact_id]
        s3[("theseus-models", storage.export_key(run, artifact.filename))] = b"CONVERTED"

    monkeypatch.setattr(export_job, "_convert", fake_convert)
    await export_job.run_export(export_id)

    e = await row(db, export_id)
    assert e.status == "ready"
    with zipfile.ZipFile(io.BytesIO(s3[("theseus-models", e.bundle_key)])) as zf:
        assert zf.read("model.onnx") == b"CONVERTED"


def test_export_ids_are_uuids_in_bundle_keys():
    assert storage.bundle_key("run", str(uuid.UUID(int=1))).endswith("00000000-0000-0000-0000-000000000001.zip")
