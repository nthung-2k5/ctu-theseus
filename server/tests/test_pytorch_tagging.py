"""Token classification in the PyTorch backend: one tag per whitespace-separated word, from the word ids to the
entity-level report, the ONNX graph and the Python client."""

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import omnitrain.plugins.export
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor
from omnitrain.domain.token_report import entities, is_bio
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, TrainContext
from omnitrain.plugins.backends.pytorch import PytorchBackend, network, tagging, tasks
from omnitrain.plugins.backends.pytorch import train as train_mod
from omnitrain.plugins.backends.pytorch.compile import (
    PytorchHyperparameters,
    compile_pytorch_config,
    hyperparameter_specs,
)
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.spec import TextSpec, WordsSpec
from omnitrain.plugins.backends.registry import get_backend, trainable_backends
from omnitrain.plugins.export.registry import list_export_formats
from omnitrain.scoring.base import list_scores

TASK = get_task_descriptor("token_classification")
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="text", kind="inline_text"),
        ColumnSpec(name="tags", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=[],
)
CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"


class TestWordIds:
    def test_only_ascii_capitals_are_lowered_and_a_word_is_never_split(self):
        assert tagging.word_forms("Mr. ĐÀ-Nẵng, ABC\tx") == ["mr.", "ĐÀ-nẵng,", "abc", "x"]

    def test_the_vocabulary_is_most_frequent_first_alphabetical_among_equals_and_has_a_floor(self):
        texts = ["b a b c", "a b d", "A e"]
        assert tagging.fit_vocabulary(texts, max_words=10, min_count=1) == ["a", "b", "c", "d", "e"]
        assert tagging.fit_vocabulary(texts, max_words=10, min_count=2) == ["a", "b"]
        assert tagging.fit_vocabulary(texts, max_words=3, min_count=1) == ["a", "b", "c"]

    def test_encoding_looks_up_pads_and_cuts(self):
        spec = TextSpec(max_tokens=4, vocab=("a", "b"))
        ids = tagging.encode(["A b zz", "", "a a a a a b"], spec)
        assert ids.tolist() == [[2, 3, 1, 0], [0, 0, 0, 0], [2, 2, 2, 2]]
        assert ids.dtype == np.int64

    def test_labels_are_ignored_where_the_model_reads_nothing(self):
        out = tagging.label_ids([["x", "y"], ["y"] * 5], {"x": 0, "y": 1}, 3)
        assert out.tolist() == [[0, 1, tagging.IGNORE], [1, 1, 1]]

    def test_the_ignore_value_is_the_losss(self):
        from omnitrain.plugins.backends.pytorch import loop

        assert tagging.IGNORE == loop.Tagging.IGNORE


def frame(rows):
    return pd.DataFrame(rows, columns=["text", "tags", "split"])


class TestPrepare:
    def test_only_rows_with_one_tag_per_word_are_used_and_the_tag_set_puts_o_first(self):
        df = frame(
            [
                ("Anna lives in Hue", "B-PER O O B-LOC", "train"),
                ("Bob", "B-PER", "train"),
                ("two words", "O", "train"),  # one tag short
                ("", "O", "train"),  # nothing to tag
                (None, None, "train"),
                ("no tags here", None, "train"),
                ("Hue", "B-LOC", "validation"),
                ("Hue city", "I-ORG O", "test"),  # a tag only the test split has
            ]
        )
        splits, names = tagging.prepare(df, text_column="text", tags_column="tags", split_column="split")
        assert names == ["O", "B-LOC", "B-PER", "I-ORG"]
        assert [len(splits[s]) for s in ("train", "validation", "test")] == [2, 1, 1]
        assert splits["train"].tags[0] == ["B-PER", "O", "O", "B-LOC"]


class TestEntities:
    @pytest.mark.parametrize(
        ("tags", "expected"),
        [
            (["B-PER", "I-PER", "O", "B-LOC"], {("PER", 0, 1), ("LOC", 3, 3)}),
            (["B-PER", "B-PER"], {("PER", 0, 0), ("PER", 1, 1)}),
            (["O", "I-PER", "I-PER"], {("PER", 1, 2)}),  # a stray I- starts an entity
            (["B-PER", "I-LOC"], {("PER", 0, 0), ("LOC", 1, 1)}),  # a change of type is a new entity
            (["B-PER", "O", "I-PER"], {("PER", 0, 0), ("PER", 2, 2)}),
            (["O", "O"], set()),
            (["PER", "LOC"], set()),  # not BIO: no entities
            ([], set()),
        ],
    )
    def test_spans(self, tags, expected):
        assert entities(tags) == expected

    def test_a_tag_set_is_bio_when_it_has_an_entity_tag(self):
        assert is_bio(["O", "B-PER"]) and is_bio(["I-X"])
        assert not is_bio(["O", "NOUN", "VERB"]) and not is_bio(["B-", "O"])


class TestObjective:
    def test_positions_without_a_label_count_for_nothing(self):
        import torch

        from omnitrain.plugins.backends.pytorch import loop

        torch.manual_seed(0)
        outputs = torch.randn(2, 3, 4)
        targets = torch.tensor([[1, 2, loop.Tagging.IGNORE], [0, loop.Tagging.IGNORE, loop.Tagging.IGNORE]])
        objective = loop.Tagging()
        flat = torch.nn.functional.cross_entropy(outputs[targets != -100], targets[targets != -100], reduction="mean")
        assert objective.loss(outputs, targets).item() == pytest.approx(flat.item(), rel=1e-5)
        sums = objective.sums(outputs, targets)
        assert sums["count"] == 3 and sums["loss"] == pytest.approx(flat.item() * 3, rel=1e-5)
        metrics = objective.summarise({"loss": 6.0, "correct": 2.0, "count": 3.0}, seen=2)
        assert metrics == {"loss": 2.0, "accuracy": pytest.approx(2 / 3)}

    def test_a_batch_with_nothing_to_score_is_zero_not_nan(self):
        import torch

        from omnitrain.plugins.backends.pytorch import loop

        outputs, targets = torch.randn(2, 3, 4), torch.full((2, 3), loop.Tagging.IGNORE)
        assert loop.Tagging().loss(outputs, targets).item() == 0.0
        assert loop.Tagging(torch.ones(4)).loss(outputs, targets).item() == 0.0

    def test_weights_steer_the_loss_but_not_the_reported_numbers(self):
        import torch

        from omnitrain.plugins.backends.pytorch import loop

        torch.manual_seed(1)
        outputs, targets = torch.randn(3, 4, 3), torch.randint(0, 3, (3, 4))
        plain, weighted = loop.Tagging(), loop.Tagging(torch.tensor([0.1, 1.0, 5.0]))
        assert weighted.loss(outputs, targets).item() != pytest.approx(plain.loss(outputs, targets).item())
        assert weighted.sums(outputs, targets) == plain.sums(outputs, targets)


class TestNetwork:
    def test_it_gives_one_row_of_logits_per_position(self):
        import torch

        net = network.build_tagger("tagger_cnn", 30, 5, 0.1)
        assert net(torch.randint(0, 30, (3, 12))).shape == (3, 12, 5)

    def test_how_far_a_text_is_padded_changes_nothing(self):
        import torch

        torch.manual_seed(0)
        net = network.build_tagger("tagger_cnn", 12, 4, 0.0).eval()
        spec = TextSpec(max_tokens=8, vocab=tuple("abcdefgh"))
        text = ["a b zz c d"]
        short = torch.from_numpy(tagging.encode(text, spec))
        long = torch.from_numpy(tagging.encode(text, TextSpec(max_tokens=30, vocab=spec.vocab)))
        with torch.no_grad():
            a, b = net(short), net(long)
        torch.testing.assert_close(a, b[:, :8], rtol=1e-5, atol=1e-5)

    def test_a_word_is_tagged_by_its_neighbours_not_only_itself(self):
        import torch

        torch.manual_seed(0)
        net = network.build_tagger("tagger_cnn", 12, 4, 0.0).eval()
        base = torch.arange(2, 14).reshape(1, 12) % 10 + 2
        changed = base.clone()
        changed[0, 3] = 11
        with torch.no_grad():
            a, b = net(base), net(changed)
        assert not torch.allclose(a[0, 2], b[0, 2]) and not torch.allclose(a[0, 6], b[0, 6])  # within four words
        # Two layers of width five see four words either side, no further: the far end is untouched.
        assert torch.allclose(a[0, 8:], b[0, 8:])

    def test_unknown_models_are_refused(self):
        with pytest.raises(ValueError, match="not a token classification model"):
            network.build_tagger("text_bag", 10, 3, 0.1)


class TestCompile:
    def test_the_config_describes_a_tagger(self):
        config = compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters())
        assert (config["inputKind"], config["task"], config["model"]) == ("words", "tag", "tagger_cnn")
        assert config["textColumn"] == "text" and config["labelColumn"] == "tags"
        # The tag set is not known to the compiler: a snapshot of tags has no class list.
        assert config["classNames"] == []
        assert config["text"]["maxTokens"] == tasks.TEXT_DEFAULTS["max_tokens"]

    def test_knobs_of_other_inputs_and_pretrained_weights_are_refused(self):
        for bad in ({"image_size": 64}, {"sample_rate": 8000}, {"horizontal_flip": True}):
            with pytest.raises(ConfigError, match="token_classification"):
                compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(**bad))
        with pytest.raises(ConfigError, match="trained from scratch"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(freeze_backbone=True))
        with pytest.raises(ConfigError, match="not a token classification model"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(model_id="text_bag"))

    def test_only_the_token_kind_of_bring_your_own_model_is_accepted(self):
        ref = CustomModelRef(id="u1", kind="hf_text", local_path="/x", source_kind="upload")
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(), custom=ref)

    def test_the_backend_offers_the_task_and_its_form(self):
        assert PytorchBackend.supports(TASK)
        assert [m.id for m in PytorchBackend.models(TASK)] == ["tagger_cnn"]
        names = {p.name for p in hyperparameter_specs(TASK)}
        assert {"epochs", "maxTokens", "vocabSize", "useClassWeights", "dropout"} <= names
        assert "pytorch" in [b.id for b in trainable_backends(TASK)]

    def test_entity_f1_is_the_headline_before_accuracy(self):
        scores = [s.id for s in list_scores()]
        assert scores.index("entityF1") < scores.index("accuracy")


# -- Training ------------------------------------------------------------------------------------------------------

PEOPLE = ["Anna", "Binh", "Chau", "Dung", "Emma", "Farid"]
CITIES = ["Hue", "Hanoi", "Paris", "Lyon", "Oslo", "Rome"]
RARE_PEOPLE = ["Zorblax", "Quenby", "Vortigern", "Ulrika"]  # each seen once in training: read as unknown


def sentence(rng, person, city):
    """Three templates; the tag is given by the words around the name, so an unknown name is still found."""
    kind = int(rng.integers(0, 3))
    if kind == 0:
        words = ["Mr.", person, "lives", "in", city, "."]
        tags = ["O", "B-PER", "O", "O", "B-LOC", "O"]
    elif kind == 1:
        words = [person, "Smith", "visited", city, "yesterday"]
        tags = ["B-PER", "I-PER", "O", "B-LOC", "O"]
    else:
        words = ["Today", "Dr.", person, "left", city]
        tags = ["O", "O", "B-PER", "O", "B-LOC"]
    return " ".join(words), " ".join(tags)


def make_corpus(rng) -> pd.DataFrame:
    rows = []
    for split, count in (("train", 160), ("validation", 40), ("test", 40)):
        for i in range(count):
            person = PEOPLE[int(rng.integers(len(PEOPLE)))]
            if split != "train" and i % 2 == 0:
                person = RARE_PEOPLE[int(rng.integers(len(RARE_PEOPLE)))]
            text, tags = sentence(rng, person, CITIES[int(rng.integers(len(CITIES)))])
            rows.append({"text": text, "tags": tags, "split": split, "_theseus_item_id": f"{split}-{i}"})
    for k, person in enumerate(RARE_PEOPLE):
        text, tags = sentence(rng, person, "Hue")
        rows.append({"text": text, "tags": tags, "split": "train", "_theseus_item_id": f"rare-{k}"})
    # Rows that cannot be used: no tags, a tag short, blank text.
    rows += [
        {"text": "Mr. Anna lives", "tags": None, "split": "train", "_theseus_item_id": "bad-0"},
        {"text": "Mr. Anna lives", "tags": "O B-PER", "split": "train", "_theseus_item_id": "bad-1"},
        {"text": "", "tags": "O", "split": "train", "_theseus_item_id": "bad-2"},
    ]
    return pd.DataFrame(rows)


def train_tagger(tmp_path, monkeypatch, df=None, **selection):
    df = make_corpus(np.random.default_rng(0)) if df is None else df
    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    config = compile_pytorch_config(
        TASK, SNAPSHOT,
        PytorchHyperparameters(**{"seed": 1, "epochs": 25, "batch_size": 16, "min_word_count": 2, **selection}),
    )  # fmt: skip
    reports: list = []
    ctx = TrainContext(
        run_id="tags", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )  # fmt: skip
    model = PytorchBackend.train(ctx)
    return types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, config=config)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        return train_tagger(tmp_path_factory.mktemp("tags"), mp)


def test_training_reports_per_word_numbers_and_uploads_the_model(trained, tmp_path):
    assert sorted(p.name for p in trained.uploaded.iterdir()) == ["model.pt", "names.json"]
    assert all({"loss", "accuracy"} <= set(m) for _, _, m in trained.reports)
    assert max(m["accuracy"] for _, s, m in trained.reports if s == "validation") > 0.95

    meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
    assert (meta["task"], meta["inputKind"], meta["model"]) == ("tag", "words", "tagger_cnn")
    # The model's own tag order, read off the data: `O` first, then alphabetical.
    assert meta["names"] == ["O", "B-LOC", "B-PER", "I-PER"]
    spec = WordsSpec.from_meta(meta)
    # Built from the train texts alone: an unseen name is not in it, a name seen once is below the floor.
    assert "anna" in spec.vocab and "mr." in spec.vocab
    assert not {"zorblax", "quenby"} & set(spec.vocab)


def test_predict_gives_one_tag_per_word_and_finds_names_it_never_saw(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    assert loaded.output.kind == "tokens" and loaded.output.labels == ["O", "B-LOC", "B-PER", "I-PER"]
    assert loaded.input_columns == ["text"]

    text = "Mr. Xanthippe lives in Paris ."
    predictions = loaded.predict(pd.DataFrame({"text": [text, "", "Dr. Anna left Lyon"]}))
    assert predictions["tags_predictions"].tolist()[0] == "O B-PER O O B-LOC O"
    assert predictions["tags_predictions"].tolist()[1] == ""
    assert predictions["tags_words"].tolist()[2] == ["Dr.", "Anna", "left", "Lyon"]
    assert predictions["tags_predictions"].tolist()[2].split()[1:2] == ["B-PER"]
    assert predictions["tags_probability"].tolist()[1] == 1.0

    output = loaded.to_output(predictions.iloc[[0]], input_tokens=text.split())
    assert output["kind"] == "tokens"
    assert [t["tag"] for t in output["tokens"]] == ["O", "B-PER", "O", "O", "B-LOC", "O"]
    assert [t["token"] for t in output["tokens"]] == text.split()
    # A tagging has no distribution over classes: the golden sample carries the tags instead.
    assert loaded.golden_prediction(predictions.iloc[[0]]) == {}


def test_a_word_past_the_reading_limit_gets_the_outside_tag_and_no_confidence(tmp_path, monkeypatch):
    run = train_tagger(tmp_path, monkeypatch, epochs=3, max_tokens=8)
    text = "Mr. Anna lives in Hue . " + "word " * 9 + "end"
    predictions = run.model.predict(pd.DataFrame({"text": [text]}))
    tags = predictions["tags_predictions"].iloc[0].split()
    assert len(tags) == len(text.split()) == 16
    assert tags[8:] == ["O"] * 8
    confidences = predictions["tags_word_probabilities"].iloc[0]
    assert len(confidences) == 16 and confidences[8:] == [0.0] * 8
    assert predictions["tags_probability"].iloc[0] == 0.0


def test_evaluation_counts_words_and_whole_entities(trained):
    result = PytorchBackend.evaluate(
        PytorchBackend.load(str(trained.uploaded)), trained.df, "split", "_theseus_item_id"
    )
    assert result is not None
    report = result.report
    assert (report["split"], report["outputType"], report["unit"]) == ("test", "tags", "word")
    # 40 test sentences; every one has a name the model never saw in half of them.
    assert report["rowCount"] == 40
    assert report["idx2str"] == ["O", "B-LOC", "B-PER", "I-PER"]
    overall = report["overall"]
    assert overall["accuracy"] > 0.95 and overall["entityF1"] > 0.9
    assert {"entityPrecision", "entityRecall", "macroF1"} <= set(overall)
    matrix = np.array(report["confusionMatrix"])
    assert matrix.shape == (4, 4) and matrix.sum() == report["wordCount"]
    assert set(report["perClass"]) == set(report["idx2str"])
    assert set(result.predictions.columns) >= {"tags_predictions", "tags_probability", "tags_actual"}

    from omnitrain.services.evaluation_score import score_from_overall

    score = score_from_overall(overall)
    assert score is not None and (score.metric, score.label, score.value) == (
        "entityF1", "Entity F1", overall["entityF1"],
    )  # fmt: skip


def test_the_report_is_wrong_in_the_places_the_model_is_wrong(tmp_path, monkeypatch):
    """A model trained for one epoch on a hard split must score below a trained one, and an all-`O` guess scores well
    on words and zero on entities: the numbers can tell good from bad."""
    from omnitrain.domain.token_report import evaluate_tokens

    class Everything(types.SimpleNamespace):
        def predict(self, frame):
            n = frame["text"].str.split().str.len()
            return pd.DataFrame(
                {"tags_predictions": [" ".join(["O"] * k) for k in n], "tags_probability": [0.9] * len(n)}
            )

    df = make_corpus(np.random.default_rng(0))
    model = Everything(
        output=types.SimpleNamespace(labels=["O", "B-LOC", "B-PER", "I-PER"], name="tags"), input_columns=["text"]
    )
    result = evaluate_tokens(model, df, "split", "_theseus_item_id", label_column="tags", batch_size=16,
                             max_classes_for_matrix=200)  # fmt: skip
    assert result is not None
    overall = result.report["overall"]
    assert 0.4 < overall["accuracy"] < 0.7 and overall["entityF1"] == 0.0
    assert len(result.report["topErrors"]) == 40


def test_a_snapshot_with_one_tag_is_refused(tmp_path, monkeypatch):
    df = make_corpus(np.random.default_rng(0))
    df["tags"] = df["text"].map(lambda t: " ".join(["O"] * len(t.split())))
    with pytest.raises(ConfigError, match="only one tag"):
        train_tagger(tmp_path, monkeypatch, df=df)


def test_a_missing_split_is_refused(tmp_path, monkeypatch):
    df = make_corpus(np.random.default_rng(0))
    with pytest.raises(ConfigError, match="validation split has no usable texts"):
        train_tagger(tmp_path, monkeypatch, df=df[df["split"] != "validation"])


def test_class_weights_train_too(tmp_path, monkeypatch):
    run = train_tagger(tmp_path, monkeypatch, use_class_weights=True, epochs=8)
    assert max(m["accuracy"] for _, s, m in run.reports if s == "validation") > 0.9


# -- Export --------------------------------------------------------------------------------------------------------


def load_client(model_path, meta_path):
    module = types.ModuleType("omnitrain_client_tags")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    return module.OmniTrainClient(model_path, meta_path)


def test_the_manifest_describes_the_recipe_and_the_tags(trained):
    meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
    manifest = build_preprocessing_manifest("tags", {}, meta)
    (output,) = manifest["outputs"]
    assert output["type"] == "tags" and output["classes"] == meta["names"]
    (input_spec,) = manifest["inputs"]
    recipe = input_spec["tagWords"]
    assert input_spec["type"] == "text" and "textTokens" not in input_spec
    ids = (recipe["recipe"], recipe["padId"], recipe["unknownId"], recipe["firstWordId"])
    assert ids == ("tag_word_ids_v1", 0, 1, 2)
    assert recipe["fallbackTag"] == "O" and recipe["vocabulary"] == meta["vocabulary"]


def test_the_onnx_graph_and_the_shipped_client_agree_with_the_platform(trained, tmp_path):
    import onnxruntime as ort
    import torch

    from omnitrain.plugins.backends.pytorch import runtime

    loaded = PytorchBackend.load(str(trained.uploaded))
    path = PytorchBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]

    spec = WordsSpec.from_meta(loaded.meta)
    texts = ["Mr. Anna lives in Paris .", "Dr. Zzyzx left Rome", "Binh Smith visited Oslo yesterday", ""]
    ids = tagging.encode(texts, spec)
    with torch.no_grad():
        expected = loaded.net(torch.from_numpy(ids).to(runtime.device())).cpu().numpy()
    session = ort.InferenceSession(path)
    assert [(i.name, i.type) for i in session.get_inputs()] == [("tokens", "tensor(int64)")]
    (logits,) = session.run(["logits"], {"tokens": ids})
    assert logits.shape == (4, spec.max_tokens, 4)
    np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-3)
    (one,) = session.run(["logits"], {"tokens": ids[:1]})  # a batch of one: the graph was traced with two rows
    np.testing.assert_allclose(one, expected[:1], rtol=1e-3, atol=1e-3)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copyfile(path, bundle / "model.onnx")
    manifest = build_preprocessing_manifest("tags", {}, loaded.meta)
    (bundle / "preprocessing.json").write_text(json.dumps(manifest))
    client = load_client(bundle / "model.onnx", bundle / "preprocessing.json")
    for text in [*texts, "MR. ANNA lives in PARIS", "Mr. Anna " + "x " * 200 + "Paris"]:
        from_client = client.predict(text)
        from_platform = loaded.predict(pd.DataFrame({"text": [text]}))
        assert [w["tag"] for w in from_client] == from_platform["tags_predictions"].iloc[0].split()
        assert [w["token"] for w in from_client] == text.split()
        np.testing.assert_allclose(
            [w["confidence"] for w in from_client],
            from_platform["tags_word_probabilities"].iloc[0],
            atol=2e-3,
        )


class TestBundle:
    FORMATS = ("python_devkit", "typescript_devkit", "csharp_devkit", "java_devkit", "pwa_app", "flutter_app")

    @staticmethod
    def files(trained, tmp_path, fmt: str, monkeypatch) -> dict:
        from omnitrain.jobs import export as export_job
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, resolve_sample_file, to_bundle_local

        # The real golden-sample builder, over the real uploaded model and an in-memory snapshot.
        monkeypatch.setattr(export_job.storage, "s3fs_path", lambda bucket, key: "snapshot")
        monkeypatch.setattr(export_job.pd, "read_parquet", lambda path: trained.df)
        golden_json = export_job._build_golden_sample(get_backend("pytorch"), str(trained.uploaded), "key")
        assert golden_json is not None and golden_json["outputType"] == "tokens" and golden_json["predictions"] == {}

        loaded = PytorchBackend.load(str(trained.uploaded))
        onnx_path = get_backend("pytorch").convert(loaded, "onnx", str(tmp_path / f"export-{fmt}"))
        manifest = build_preprocessing_manifest("tags", {}, loaded.meta)
        filename, data = resolve_sample_file(golden_json["inputValue"], None)
        golden = GoldenSample(to_bundle_local(golden_json, filename), filename, data)
        made = assemble_files(
            run_name="r", task_label="Token Classification", format_id=fmt,
            model_bytes=Path(onnx_path).read_bytes(), preprocessing=manifest, golden=golden,
        )  # fmt: skip
        return {f.path: f.data for f in made}

    def test_every_format_is_offered_the_model_and_only_python_implements_it(self, trained, tmp_path, monkeypatch):
        assert {f.id for f in list_export_formats(TASK, PytorchBackend)} >= set(self.FORMATS)
        for fmt in self.FORMATS[1:]:
            readme = self.files(trained, tmp_path, fmt, monkeypatch)["README.md"].decode()
            assert "tags every word" in readme and "does not implement" in readme, fmt

    def test_the_golden_sample_carries_the_tags_and_the_python_bundle_verifies_them(
        self, trained, tmp_path, monkeypatch
    ):
        files = self.files(trained, tmp_path, "python_devkit", monkeypatch)
        expected = json.loads(files["expected.json"])
        assert expected["predictions"] == {} and expected["outputType"] == "tokens"
        sample = files["sample/input.txt"].decode()
        assert len(expected["tags"]) == len(sample.split())
        readme = files["README.md"].decode()
        assert "Token classification" in readme and "[{token, tag, confidence}]" in readme

        out = tmp_path / "bundle"
        for path, data in files.items():
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        ran = subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 0 and ran.stdout.startswith("OK"), ran.stdout + ran.stderr

        # A wrong tag in expected.json is caught, naming the word.
        expected["tags"][0] = "I-PER" if expected["tags"][0] != "I-PER" else "O"
        (out / "expected.json").write_text(json.dumps(expected))
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 1 and "MISMATCH word 0" in ran.stdout

        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "example.py", "Mr. Anna lives in Paris ."],
            cwd=out, capture_output=True, text=True, timeout=300,
        )  # fmt: skip
        lines = [line.split("\t") for line in ran.stdout.splitlines()]
        assert ran.returncode == 0 and [line[:2] for line in lines][:2] == [["Mr.", "O"], ["Anna", "B-PER"]], ran.stderr
