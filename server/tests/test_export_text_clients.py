"""The generated Python devkit client for a text classifier, and what every generated client ships for one.

The model is built by hand (an embedding table, a mean over the ids, a linear map), so what is under test is the
client's own arithmetic: the tokenizer and the vocabulary lookup, each against the backend's reference in
`plugins/backends/pytorch/text.py`, which is what the manifest's `textTokens` describes.
"""

import json
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from omnitrain.plugins.backends.pytorch import text
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.spec import TextSpec

TEMPLATES = Path(__file__).resolve().parents[1] / "omnitrain" / "plugins" / "export" / "templates" / "python"
CLASSES = ["sport", "food", "tech"]
VOCAB = ("goal", "team", "soup", "code", "việt", "nam", "đà", "nẵng", "3", "14", "ok")
SPEC = TextSpec(max_tokens=12, vocab=VOCAB)
EMBEDDING = 6


def client_module():
    source = (TEMPLATES / "omnitrain_client.py").read_text(encoding="utf-8")
    module = types.ModuleType("omnitrain_client_text_under_test")
    module.__file__ = str(TEMPLATES / "omnitrain_client.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - our own shipped template
    return module


def text_meta(spec: TextSpec = SPEC) -> dict:
    return {
        "task": "classify", "model": "text_bag", **spec.to_meta(), "textColumn": "review",
        "network": {"dropout": 0.0}, "outputName": "class", "names": CLASSES,
    }  # fmt: skip


def weights(spec: TextSpec = SPEC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    table = rng.normal(0, 1, (spec.size, EMBEDDING)).astype(np.float32)
    return (
        table,
        rng.normal(0, 1, (EMBEDDING, len(CLASSES))).astype(np.float32),
        rng.normal(0, 1, len(CLASSES)).astype(np.float32),
    )


def write_model(path: Path, spec: TextSpec = SPEC) -> Path:
    """`tokens` [batch, maxTokens] int64 -> embeddings -> mean over every position (padding included) -> a linear
    layer: logits [batch, classes]. Sensitive to every id it is given, in any position."""
    table, w, b = weights(spec)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "tokens"], ["embedded"], axis=0),
            helper.make_node("ReduceMean", ["embedded"], ["pooled"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["scores"]),
            helper.make_node("Add", ["scores", "bias"], ["logits"]),
        ],
        "tiny_text",
        [helper.make_tensor_value_info("tokens", TensorProto.INT64, ["batch", spec.max_tokens])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)])],
        [
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(w, "weights"),
            numpy_helper.from_array(b, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def write_manifest(path: Path, manifest: dict | None = None) -> Path:
    path.write_text(json.dumps(manifest or build_preprocessing_manifest("run-1", {}, text_meta())), encoding="utf-8")
    return path


def make_client(tmp_path: Path):
    return client_module().OmniTrainClient(
        write_model(tmp_path / "model.onnx"), write_manifest(tmp_path / "preprocessing.json")
    )


def reference_probabilities(value: str, spec: TextSpec = SPEC) -> list[float]:
    """What the backend's own text code and the same model give: the contract the client must reproduce."""
    table, w, b = weights(spec)
    ids = text.encode([value], spec)
    logits = table[ids[0]].mean(axis=0) @ w + b
    e = np.exp(logits - logits.max())
    return list(e / e.sum())


SAMPLES = [
    "The goal of the team",
    "GOAL Goal goal",
    "Việt Nam, Đà Nẵng! 3.14 ok",
    "ĐÀ NẴNG và VIỆT NAM",
    "don't stop_now",
    "tab\tnewline\ncarriage\rreturn",
    "no break　ideographic em narrow",
    "zero​width joins",
    "emoji \U0001f600 here goal \U0001f600goal",
    "x" * 500,
    " ".join(["goal", "team", "soup", "code"] * 10),
    "",
    "   \t\n  ",
    ",.;:!?()[]{}<>",
    "décomposé é vs é",
]


class TestTokenizing:
    @pytest.mark.parametrize("value", SAMPLES)
    def test_the_ids_are_the_backends(self, tmp_path, value):
        client = make_client(tmp_path)
        got = client._text_tokens(value)  # noqa: SLF001
        assert got.dtype == np.int64 and got.shape == (1, SPEC.max_tokens)
        np.testing.assert_array_equal(got, text.encode([value], SPEC))

    def test_random_text_over_every_kind_of_character_agrees_with_the_backend(self, tmp_path):
        client = make_client(tmp_path)
        rng = np.random.default_rng(11)
        pool = list("abcXYZ019 .,;:!?'\"-_/\\()[]{}<>@#$%^&*+=~`|\t\n\r") + [
            "à", "Đ", "ệ", "ß", "İ", " ", " ", " ", "​", " ", "　", "﻿",
            "\U0001f600", "中", "ж",
        ]  # fmt: skip
        words = list(VOCAB) + ["goal.", "Team", "ĐÀ"]
        for _ in range(300):
            parts = [str(rng.choice(pool)) if rng.random() < 0.6 else str(rng.choice(words)) for _ in range(30)]
            value = "".join(parts)
            np.testing.assert_array_equal(client._text_tokens(value), text.encode([value], SPEC), err_msg=repr(value))  # noqa: SLF001

    def test_a_word_outside_the_vocabulary_is_unknown_and_the_end_is_padding(self, tmp_path):
        got = make_client(tmp_path)._text_tokens("Goal MYSTERY soup")  # noqa: SLF001
        assert got[0].tolist() == [2, 1, 4] + [0] * 9


class TestPredict:
    @pytest.mark.parametrize("value", SAMPLES)
    def test_the_probabilities_match_the_backend(self, tmp_path, value):
        got = make_client(tmp_path).predict(value, top_k=3)
        expected = dict(zip(CLASSES, reference_probabilities(value), strict=True))
        assert set(got) == set(CLASSES)
        for label, probability in got.items():
            assert probability == pytest.approx(expected[label], abs=2e-4)

    def test_the_answer_is_sorted_and_top_k_limits_it(self, tmp_path):
        got = make_client(tmp_path).predict("goal team code", top_k=2)
        assert len(got) == 2 and list(got.values()) == sorted(got.values(), reverse=True)

    def test_an_empty_text_is_finite(self, tmp_path):
        got = make_client(tmp_path).predict("", top_k=3)
        assert all(np.isfinite(p) for p in got.values())

    def test_a_string_is_the_text_never_a_path_to_one(self, tmp_path):
        # A file with this name holds different words; the client classifies the string it was given.
        (tmp_path / "goal.txt").write_text("soup soup soup")
        got = make_client(tmp_path).predict(str(tmp_path / "goal.txt"), top_k=3)
        expected = dict(zip(CLASSES, reference_probabilities(str(tmp_path / "goal.txt")), strict=True))
        for label, probability in got.items():
            assert probability == pytest.approx(expected[label], abs=2e-4)

    def test_an_unknown_recipe_is_refused(self, tmp_path):
        manifest = build_preprocessing_manifest("run-1", {}, text_meta())
        manifest["inputs"][0]["textTokens"]["recipe"] = "word_ids_v9"
        with pytest.raises(NotImplementedError, match="word_ids_v9"):
            client_module().OmniTrainClient(
                write_model(tmp_path / "model.onnx"), write_manifest(tmp_path / "preprocessing.json", manifest)
            )

    def test_a_text_input_without_a_recipe_still_says_what_to_do(self, tmp_path):
        manifest = build_preprocessing_manifest("run-1", {}, text_meta())
        del manifest["inputs"][0]["textTokens"]
        client = client_module().OmniTrainClient(
            write_model(tmp_path / "model.onnx"), write_manifest(tmp_path / "preprocessing.json", manifest)
        )
        with pytest.raises(NotImplementedError, match="preprocessing.json"):
            client.predict("goal")


class TestBundles:
    FORMATS = {
        "python_devkit": "omnitrain_client.py",
        "typescript_devkit": "client.ts",
        "csharp_devkit": "OmniTrainClient.cs",
        "java_devkit": "OmniTrainClient.java",
        "pwa_app": "app.js",
        "flutter_app": "lib/omnitrain_client.dart",
    }

    def test_every_client_offers_text_classification(self):
        from omnitrain.domain.task_registry import get_task_descriptor
        from omnitrain.plugins.export.registry import list_export_formats

        task = get_task_descriptor("text_classification")
        assert set(self.FORMATS) <= {f.id for f in list_export_formats(task, None)}

    @staticmethod
    def bundle(fmt: str, sample: tuple[str, bytes] = ("input.txt", b"goal")) -> dict[str, str]:
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        manifest = json.loads(json.dumps(build_preprocessing_manifest("r", {}, text_meta())))
        files = assemble_files(
            run_name="r", task_label="Text Classification", format_id=fmt, model_bytes=b"m", preprocessing=manifest,
            golden=GoldenSample(b"{}", *sample),
        )  # fmt: skip
        return {f.path: f.data.decode(errors="replace") for f in files}

    @pytest.mark.parametrize(("fmt", "client_file"), list(FORMATS.items()))
    def test_every_clients_bundle_ships_the_text_code_and_says_so(self, fmt, client_file):
        files = self.bundle(fmt)
        client = files[client_file]
        for needed in ("textTokens", "word_ids_v1", "separatorRanges", "unknownId", "maxTokens"):
            assert needed.lower() in client.lower(), needed
        readme = files["README.md"]
        assert "Text classification" in readme
        assert "image classification** inputs today" not in readme

    def test_a_scikit_learn_text_model_keeps_the_old_behaviour(self):
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files

        manifest = {
            "schemaVersion": 1,
            "runId": "r",
            "inputs": [{"name": "text", "type": "text", "column": "text", "textInput": {"encoding": "utf-8"}}],
            "outputs": [{"name": "class", "type": "category", "column": "class", "classes": ["a", "b"]}],
        }
        files = assemble_files(
            run_name="r", task_label="Text Classification", format_id="python_devkit", model_bytes=b"m",
            preprocessing=manifest, golden=GoldenSample(b"{}", "input.txt", b"x"),
        )  # fmt: skip
        readme = {f.path: f.data.decode() for f in files}["README.md"]
        assert "Text classification: the model takes word ids" not in readme

    def test_the_python_bundle_verifies_a_real_sample_end_to_end(self, tmp_path):
        import subprocess
        import sys

        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, to_bundle_local

        value = "Việt Nam, Đà Nẵng: the goal of the team? 3.14"
        probabilities = {k: float(v) for k, v in zip(CLASSES, reference_probabilities(value), strict=True)}
        expected = {"outputColumn": "class", "outputType": "category", "predictions": probabilities}
        golden = GoldenSample(to_bundle_local(expected, "input.txt"), "input.txt", value.encode("utf-8"))
        write_model(tmp_path / "model.onnx")
        manifest = json.loads(json.dumps(build_preprocessing_manifest("r", {}, text_meta())))
        files = assemble_files(
            run_name="r", task_label="t", format_id="python_devkit", model_bytes=(tmp_path / "model.onnx").read_bytes(),
            preprocessing=manifest, golden=golden,
        )  # fmt: skip
        out = tmp_path / "bundle"
        for f in files:
            target = out / f.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.data)
        ran = subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 0 and ran.stdout.startswith("OK"), ran.stdout + ran.stderr

        # The same bundle with a changed prediction fails.
        (out / "expected.json").write_text(
            json.dumps(
                {
                    **json.loads((out / "expected.json").read_text()),
                    "predictions": {"sport": 0.01, "food": 0.01, "tech": 0.98},
                }
            )
        )
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=120
        )
        assert ran.returncode == 1 and "MISMATCH" in ran.stdout
