"""The generated clients for a token classifier (the built-in tagger, recipe `tag_word_ids_v1`).

The model is built by hand (an embedding per word id, a linear layer, a bias and a position term), so what is under test
is the client's own arithmetic: cutting a text into words at the manifest's whitespace, the vocabulary lookup, and the
tag of each word's row of logits. The reference is computed here from the platform's own functions
(`domain/token_tags.words`, `tagging.encode`) and numpy, never by a generated client.

Only the Python client is run here. The other five were run against the same cases (TypeScript with bun and
onnxruntime-node, C# and Java with their onnxruntime packages, the PWA in Chrome; the Flutter front end in a Dart test,
its ONNX session not run); this file holds the cheap checks that they all ship the recipe and say so.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from omnitrain.domain.token_tags import words
from omnitrain.plugins.backends.pytorch import tagging
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.spec import WordsSpec

TEMPLATES = Path(__file__).resolve().parents[1] / "omnitrain" / "plugins" / "export" / "templates" / "python"
TAGS = ["O", "B-PER", "I-PER", "B-LOC"]
VOCAB = ("hanoi", "mr.", "paris", "việt", "nam", "đà", "nẵng", "a​b", "😀", "3", "ok", "the")
SPEC = WordsSpec(max_tokens=6, vocab=VOCAB)
EMBEDDING = 6

# Whitespace of every kind `str.split()` cuts at, a zero-width space it does not, an astral character, capitals, and a
# text longer than `max_tokens` words.
TEXTS = [
    "Mr. Hanoi lives in Paris",
    "HANOI hanoi Hanoi hAnOi",
    "Việt Nam , Đà Nẵng 3 ok",
    "the\tof\nin Paris　nam ok the",
    "a​b 😀 3 😀x ok",
    "one two three four five six seven eight nine ten",
    "   ",
    "",
    "  leading and trailing  ",
]


def tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    table = rng.normal(0, 1, (SPEC.size, EMBEDDING)).astype(np.float32)
    w = rng.normal(0, 1, (EMBEDDING, len(TAGS))).astype(np.float32) * 0.9
    b = rng.normal(0, 1, len(TAGS)).astype(np.float32) * 0.5
    pos = rng.normal(0, 0.8, (SPEC.max_tokens, len(TAGS))).astype(np.float32)  # makes the position matter
    return table, w, b, pos


def write_model(path: Path) -> Path:
    table, w, b, pos = tables()
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "tokens"], ["embedded"], axis=0),
            helper.make_node("MatMul", ["embedded", "w"], ["scores"]),
            helper.make_node("Add", ["scores", "b"], ["biased"]),
            helper.make_node("Add", ["biased", "pos"], ["logits"]),
        ],
        "tiny_tagger",
        [helper.make_tensor_value_info("tokens", TensorProto.INT64, ["batch", SPEC.max_tokens])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", SPEC.max_tokens, len(TAGS)])],
        [
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(b, "b"),
            numpy_helper.from_array(pos, "pos"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def reference(text: str) -> list[dict]:
    """The platform's answer for `text`: its words and, for each, the best tag and its probability."""
    table, w, b, pos = tables()
    logits = table[tagging.encode([text], SPEC)] @ w + b + pos
    result = []
    for i, token in enumerate(words(text)):
        if i >= SPEC.max_tokens:
            result.append({"token": token, "tag": "O", "confidence": 0.0})
            continue
        row = logits[0, i].astype(np.float64)
        p = np.exp(row - row.max())
        p /= p.sum()
        k = int(np.argmax(p))
        result.append({"token": token, "tag": TAGS[k], "confidence": round(float(p[k]), 4)})
    return result


def tag_meta() -> dict:
    return {
        **SPEC.to_meta(), "task": "tag", "model": "tagger_cnn", "textColumn": "text", "network": {"dropout": 0.0},
        "outputName": "tags", "names": TAGS,
    }  # fmt: skip


def manifest() -> dict:
    return json.loads(json.dumps(build_preprocessing_manifest("r", {}, tag_meta())))


def client_module():
    source = (TEMPLATES / "omnitrain_client.py").read_text(encoding="utf-8")
    module = types.ModuleType("omnitrain_client_tags_under_test")
    module.__file__ = str(TEMPLATES / "omnitrain_client.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - our own shipped template
    return module


def bundle_files(fmt: str, model: bytes = b"m", sample: str = "Mr. Hanoi lives in Paris", tags=None):
    from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, to_bundle_local

    expected = {"outputColumn": "tags", "outputType": "tags", "predictions": {}}
    if tags is not None:
        expected["tags"] = tags
    return assemble_files(
        run_name="r", task_label="Token Classification", format_id=fmt, model_bytes=model, preprocessing=manifest(),
        golden=GoldenSample(to_bundle_local(expected, "input.txt"), "input.txt", sample.encode("utf-8")),
    )  # fmt: skip


class TestManifest:
    def test_it_carries_the_whitespace_the_clients_cut_words_at(self):
        recipe = manifest()["inputs"][0]["tagWords"]
        cut = {c for low, high in recipe["whitespaceRanges"] for c in range(low, high + 1)}
        assert cut == {c for c in range(0x10000) if chr(c).isspace()}
        assert "whitespaceRanges" in recipe["split"]


class TestPythonClient:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls, tmp_path_factory):
        folder = tmp_path_factory.mktemp("tags")
        write_model(folder / "model.onnx")
        (folder / "preprocessing.json").write_text(json.dumps(manifest()), encoding="utf-8")
        return client_module().OmniTrainClient(folder / "model.onnx", folder / "preprocessing.json")

    @pytest.mark.parametrize("text", TEXTS)
    def test_it_tags_every_word_as_the_platform_does(self, client, text):
        got = client.predict(text)
        want = reference(text)
        assert [(g["token"], g["tag"]) for g in got] == [(e["token"], e["tag"]) for e in want]
        assert max([abs(g["confidence"] - e["confidence"]) for g, e in zip(got, want, strict=True)] or [0]) < 2e-3

    def test_a_word_past_the_model_s_length_gets_the_fallback_tag_with_no_confidence(self, client):
        got = client.predict(TEXTS[5])
        assert len(got) == 10 and all(w["tag"] == "O" and w["confidence"] == 0.0 for w in got[SPEC.max_tokens :])

    def test_an_unknown_recipe_is_refused(self, tmp_path):
        write_model(tmp_path / "model.onnx")
        changed = manifest()
        changed["inputs"][0]["tagWords"]["recipe"] = "tag_word_ids_v2"
        (tmp_path / "preprocessing.json").write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(NotImplementedError, match="tag_word_ids_v2"):
            client_module().OmniTrainClient(tmp_path / "model.onnx", tmp_path / "preprocessing.json")

    def test_the_bundle_verifies_a_real_sample_end_to_end(self, tmp_path):
        text = TEXTS[0]
        files = bundle_files(
            "python_devkit", write_model(tmp_path / "m.onnx").read_bytes(), text, [w["tag"] for w in reference(text)]
        )
        out = tmp_path / "bundle"
        for f in files:
            target = out / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.data)
        ran = subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 0 and ran.stdout.startswith("OK"), ran.stdout + ran.stderr

        expected = json.loads((out / "expected.json").read_text())
        expected["tags"][0] = "I-PER" if expected["tags"][0] != "I-PER" else "B-LOC"
        (out / "expected.json").write_text(json.dumps(expected))
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 1 and "MISMATCH word 0" in ran.stdout


class TestBundles:
    FORMATS = {
        "python_devkit": "omnitrain_client.py",
        "typescript_devkit": "client.ts",
        "csharp_devkit": "OmniTrainClient.cs",
        "java_devkit": "OmniTrainClient.java",
        "pwa_app": "app.js",
        "flutter_app": "lib/omnitrain_client.dart",
    }

    def test_every_client_offers_token_classification(self):
        from omnitrain.domain.task_registry import get_task_descriptor
        from omnitrain.plugins.export.registry import list_export_formats

        task = get_task_descriptor("token_classification")
        assert set(self.FORMATS) <= {f.id for f in list_export_formats(task, None)}

    @pytest.mark.parametrize(("fmt", "client_file"), list(FORMATS.items()))
    def test_every_clients_bundle_ships_the_tag_code_and_says_so(self, fmt, client_file):
        files = {f.path: f.data.decode(errors="replace") for f in bundle_files(fmt, tags=["O"])}
        client = files[client_file].lower()
        for needed in ("tagwords", "tag_word_ids_v1", "fallbacktag", "maxtokens", "unknownid"):
            assert needed in client, needed
        if fmt != "python_devkit":  # Python's str.split() is the definition; every other language reads the table
            assert "whitespaceranges" in client
        readme = files["README.md"]
        assert "Token classification" in readme and "does not implement" not in readme
        assert "image classification** inputs today" not in readme

    @pytest.mark.parametrize("fmt", [f for f in FORMATS if f != "python_devkit"])
    def test_a_client_that_cannot_run_a_hugging_face_tagger_says_so(self, fmt):
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        hf = {
            "schemaVersion": 1,
            "runId": "r",
            "inputs": [
                {
                    "name": "ids", "type": "text", "column": "text",
                    "hfTokenizer": {"recipe": "hf_word_tokens_v1", "tokenizer": "{}", "maxTokens": 8, "padId": 0},
                }
            ],
            "outputs": [{"name": "tags", "type": "tags", "column": "tags", "classes": ["O", "B-X"]}],
        }  # fmt: skip
        files = assemble_files(
            run_name="r", task_label="Token Classification", format_id=fmt, model_bytes=b"m", preprocessing=hf,
            golden=GoldenSample(b"{}", "input.txt", b"x"),
        )  # fmt: skip
        readme = {f.path: f.data.decode() for f in files}["README.md"]
        assert "Hugging Face transformer that tags every word" in readme
