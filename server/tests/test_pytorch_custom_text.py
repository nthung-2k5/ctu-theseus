"""Bring-your-own Hugging Face text models on the PyTorch backend: kinds, file checks, compile, strict loading of a
classifier or a bare backbone, the tokenizer contract, and a real tiny train / load / predict / evaluate / ONNX round
trip. Every model is a randomly initialised few-layer network with a tiny word-level tokenizer, written to a temp
folder, so nothing is downloaded."""

import json
import shutil
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import omnitrain.plugins.export  # noqa: E402
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor  # noqa: E402
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, TrainContext  # noqa: E402
from omnitrain.plugins.backends.pytorch import PytorchBackend, custom, hf, tasks  # noqa: E402
from omnitrain.plugins.backends.pytorch import train as train_mod  # noqa: E402
from omnitrain.plugins.backends.pytorch.compile import PytorchHyperparameters, compile_pytorch_config  # noqa: E402
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest  # noqa: E402
from omnitrain.plugins.backends.pytorch.model import HFTextModel, PytorchLoadedModel  # noqa: E402
from omnitrain.plugins.backends.pytorch.spec import HFTextSpec  # noqa: E402
from omnitrain.plugins.backends.registry import get_backend  # noqa: E402

TASK = get_task_descriptor("text_classification")
CLASSES = ["negative", "positive"]
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="text", kind="inline_text"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=CLASSES,
)
CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"

WORDS = ["good", "great", "fine", "bad", "awful", "poor", "film", "movie", "plot", "the", "a", "and", "is"]
# The special tokens of each family, in id order, and the one that wraps a text: BERT-likes pad with id 0, RoBERTa-likes
# with id 1 (and number their positions from it).
BERT_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
ROBERTA_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>")
_BASE = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64}
# family -> (config class, extra config arguments, bare class, classifier class, specials)
FAMILIES = {
    "bert": ("BertConfig", {**_BASE, "max_position_embeddings": 64}, "BertModel", "BertForSequenceClassification",
             BERT_SPECIALS),
    "distilbert": (
        "DistilBertConfig",
        {"dim": 32, "n_layers": 2, "n_heads": 2, "hidden_dim": 64, "max_position_embeddings": 64},
        "DistilBertModel", "DistilBertForSequenceClassification", BERT_SPECIALS,
    ),
    "roberta": (
        "RobertaConfig",
        {**_BASE, "max_position_embeddings": 66, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2},
        "RobertaModel", "RobertaForSequenceClassification", ROBERTA_SPECIALS,
    ),
    "xlm-roberta": (
        "XLMRobertaConfig",
        {**_BASE, "max_position_embeddings": 66, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2},
        "XLMRobertaModel", "XLMRobertaForSequenceClassification", ROBERTA_SPECIALS,
    ),
    "albert": (
        "AlbertConfig", {**_BASE, "embedding_size": 16, "max_position_embeddings": 64},
        "AlbertModel", "AlbertForSequenceClassification", BERT_SPECIALS,
    ),
    "electra": (
        "ElectraConfig", {**_BASE, "embedding_size": 16, "max_position_embeddings": 64},
        "ElectraModel", "ElectraForSequenceClassification", BERT_SPECIALS,
    ),
}  # fmt: skip


def write_tokenizer(path: Path, specials: tuple[str, ...]) -> int:
    """A word-level tokenizer over `WORDS` that wraps a text in the family's first and third special tokens, saved the
    way a Hub repo carries one (tokenizer.json and tokenizer_config.json). Returns the vocabulary size."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate([*specials, *WORDS])}
    pad, unk, start, end = (
        (specials[1], specials[3], specials[0], specials[2])
        if specials == ROBERTA_SPECIALS
        else (specials[0], specials[1], specials[2], specials[3])
    )
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=unk))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start} $A {end}", special_tokens=[(start, vocab[start]), (end, vocab[end])]
    )
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token=pad, unk_token=unk).save_pretrained(path)
    return len(vocab)


def make_tiny(path: Path, family: str = "bert", *, bare: bool = False):
    """A random few-layer checkpoint of `family` with its tokenizer, as a classifier (5 labels) or a bare backbone.
    Returns the model that was saved, so a test can compare weights against it."""
    import transformers

    config_name, arguments, backbone_name, classifier_name, specials = FAMILIES[family]
    path.mkdir(parents=True, exist_ok=True)
    vocab_size = write_tokenizer(path, specials)
    config = getattr(transformers, config_name)(vocab_size=vocab_size, **arguments)
    if bare:
        model = getattr(transformers, backbone_name)(config)
    else:
        config.num_labels = 5
        model = getattr(transformers, classifier_name)(config)
    model.save_pretrained(path, safe_serialization=True)
    return model


def ref_for(path: Path, kind: str = custom.HF_TEXT) -> CustomModelRef:
    return CustomModelRef(id="u1", kind=kind, source_kind="upload", local_path=str(path))


def copy_of(source: Path, tmp_path: Path) -> Path:
    folder = tmp_path / "m"
    shutil.copytree(source, folder)
    return folder


def edit_config(folder: Path, **changes) -> None:
    config = json.loads((folder / "config.json").read_text())
    (folder / "config.json").write_text(json.dumps({**config, **changes}))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("bert")
    make_tiny(path)
    return path


class TestKindAndValidation:
    def test_the_kind_is_offered_for_text_classification_only(self):
        assert [k.id for k in PytorchBackend.custom_model_kinds(TASK)] == [custom.HF_TEXT]
        assert custom.HF_TEXT not in [
            k.id for k in PytorchBackend.custom_model_kinds(get_task_descriptor("image_classification"))
        ]
        assert PytorchBackend.custom_model_kinds(get_task_descriptor("tabular_classification")) == []

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"model_type": "bert", "architectures": ["BertForMaskedLM"]}, custom.HF_TEXT),
            ({"model_type": "roberta", "architectures": ["RobertaModel"]}, custom.HF_TEXT),
            ({"model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"]}, custom.HF_TEXT),
            ({"model_type": "bert", "architectures": ["BertForTokenClassification"]}, None),
            ({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}, None),
            ({"model_type": "vit", "architectures": ["ViTModel"]}, custom.HF_IMAGE),
        ],
    )
    def test_the_architecture_says_which_checkpoints_are_text_models(self, config, expected):
        assert PytorchBackend.suggest_custom_kind(config) == expected

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_supported_family_is_listed_and_its_model_validates(self, family, tmp_path):
        make_tiny(tmp_path, family)
        PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))

    def test_a_bare_backbone_validates_too(self, tmp_path):
        make_tiny(tmp_path, "bert", bare=True)
        PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))

    def test_every_listed_family_has_a_test_row(self):
        assert sorted(custom.SUPPORTED_TEXT_MODEL_TYPES) == sorted(FAMILIES)

    def test_an_image_kind_is_refused_for_text(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            PytorchBackend.validate_custom_model(TASK, ref_for(tiny, custom.HF_IMAGE))

    def test_code_that_runs_from_the_repo_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, auto_map={"AutoModel": "modelling.Model"})
        with pytest.raises(ConfigError, match="custom code"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_without_safetensors_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        (folder / "model.safetensors").rename(folder / "pytorch_model.bin")
        with pytest.raises(ConfigError, match="safetensors"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_of_another_family_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, model_type="gpt2")
        with pytest.raises(ConfigError, match="Not a supported text model"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_without_a_tokenizer_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (folder / name).unlink(missing_ok=True)
        with pytest.raises(ConfigError, match="cannot be loaded as a text classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_that_does_not_say_how_long_a_text_may_be_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        config = json.loads((folder / "config.json").read_text())
        del config["max_position_embeddings"]
        (folder / "config.json").write_text(json.dumps(config))
        with pytest.raises(ConfigError, match="max_position_embeddings"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_weights_that_do_not_match_the_config_are_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, hidden_size=64, intermediate_size=128)
        with pytest.raises(ConfigError, match="cannot be loaded as a text classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_tokenizer_whose_padding_id_is_not_the_models_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, pad_token_id=3)
        with pytest.raises(ConfigError, match="pads with id 0 but the model expects 3"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))


class TestTokenLimit:
    def test_it_is_the_models_positions(self, tiny):
        assert custom.text_token_limit(tiny) == 64

    def test_roberta_numbers_its_positions_after_the_padding_id(self, tmp_path):
        make_tiny(tmp_path, "roberta")
        assert custom.text_token_limit(tmp_path) == 64  # 66 positions, less the padding id plus one

    def test_a_model_that_reads_a_handful_of_positions_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, max_position_embeddings=4)
        with pytest.raises(ConfigError, match="too few positions"):
            custom.text_token_limit(folder)


class TestCompile:
    def compile(self, tiny, **selection):
        return compile_pytorch_config(
            TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1", **selection), custom=ref_for(tiny)
        )

    def test_a_custom_model_is_recorded_for_training_and_brings_its_own_vocabulary(self, tiny):
        c = self.compile(tiny, vocab_size=500, min_word_count=1, dropout=0.5)
        assert (c["inputKind"], c["task"], c["model"], c["pretrained"]) == ("text", "classify", "custom:u1", True)
        assert c["custom"] == {"kind": custom.HF_TEXT, "path": str(tiny)}
        assert c["text"] == {"maxTokens": 128}
        assert "network" not in c and c["classNames"] == CLASSES

    def test_it_is_fine_tuned_so_the_learning_rate_defaults_small(self, tiny):
        assert self.compile(tiny)["train"]["learningRate"] == tasks.HF_TEXT_LEARNING_RATE
        assert self.compile(tiny, learning_rate=0.01)["train"]["learningRate"] == 0.01

    def test_the_other_training_knobs_still_apply(self, tiny):
        c = self.compile(tiny, max_tokens=32, epochs=3, batch_size=8, freeze_backbone=True, use_class_weights=True)
        assert c["text"] == {"maxTokens": 32}
        assert (c["train"]["epochs"], c["train"]["batchSize"]) == (3, 8)
        assert c["train"]["freezeBackbone"] is True and c["train"]["useClassWeights"] is True

    def test_training_from_scratch_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="from scratch"):
            self.compile(tiny, pretrained=False)

    def test_a_custom_model_of_the_wrong_kind_is_an_error(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(
                TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1"), custom=ref_for(tiny, custom.HF_IMAGE)
            )

    def test_the_built_in_models_refuse_to_freeze_or_start_from_weights(self):
        for selection in ({"freeze_backbone": True}, {"pretrained": True}):
            with pytest.raises(ConfigError, match="trained from scratch"):
                compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(**selection))
        built_in = compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(freeze_backbone=False))
        assert "custom" not in built_in and built_in["train"]["learningRate"] == tasks.TEXT_DEFAULTS["learning_rate"]


def backbone_matches(net, original) -> int:
    """How many of the original checkpoint's backbone tensors the network holds unchanged."""
    import torch

    mine = net.model.state_dict()
    matched = 0
    for key, value in original.state_dict().items():
        for candidate in (key, f"{net.model.base_model_prefix}.{key}"):
            if candidate in mine and torch.equal(mine[candidate], value):
                matched += 1
                break
    return matched


def text_ids(path: Path, texts: list[str], max_tokens: int = 12):
    tokenizer_json, pad_id, pad_token = hf.read_tokenizer(str(path))
    spec = HFTextSpec(max_tokens=max_tokens, pad_id=pad_id, pad_token=pad_token, tokenizer=tokenizer_json)
    return spec, hf.encode_texts(hf.make_tokenizer(spec), texts)


@pytest.mark.parametrize("bare", [False, True], ids=["classifier", "bare backbone"])
@pytest.mark.parametrize("family", FAMILIES)
class TestNetwork:
    def test_the_pretrained_weights_are_really_loaded_and_the_head_is_new(self, family, bare, tmp_path):
        original = make_tiny(tmp_path, family, bare=bare)
        net = hf.build_text(str(tmp_path), CLASSES, hf.read_tokenizer(str(tmp_path))[1])
        body = [k for k in original.state_dict() if not k.startswith(("classifier.", "pre_classifier.", "pooler."))]
        assert backbone_matches(net, original) >= len(body) - 2  # the pooler of a bare backbone is the only exception
        assert net.model.config.num_labels == 2 and net.model.config.id2label == {0: "negative", 1: "positive"}

    def test_freezing_trains_only_the_head(self, family, bare, tmp_path):
        make_tiny(tmp_path, family, bare=bare)
        net = hf.build_text(str(tmp_path), CLASSES, hf.read_tokenizer(str(tmp_path))[1])
        hf.freeze_backbone(net)
        trainable = [n for n, p in net.named_parameters() if p.requires_grad]
        assert trainable and all("classifier" in n or "pooler" in n for n in trainable)

    def test_it_exports_to_one_onnx_file_that_agrees(self, family, bare, tmp_path):
        import onnxruntime as ort
        import torch

        make_tiny(tmp_path / "files", family, bare=bare)
        folder = tmp_path / "files"
        spec, ids = text_ids(folder, ["good film", "the plot is awful and poor", "", "great great great fine movie"])
        net = hf.build_text(str(folder), CLASSES, spec.pad_id)
        meta = {
            "task": "classify", "inputKind": "text", "outputName": "class", "names": CLASSES, "textColumn": "text",
            "hf": True, **spec.to_meta(),
        }  # fmt: skip
        model = HFTextModel(net, meta)
        path = get_backend("pytorch").convert(model, "onnx", str(tmp_path / "export"))
        assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
        session = ort.InferenceSession(path)
        assert [i.name for i in session.get_inputs()] == ["input_ids"]
        with torch.no_grad():
            expected = model.net(ids.to(next(model.net.parameters()).device)).cpu().numpy()
        got = session.run(None, {"input_ids": ids.numpy()})[0]
        np.testing.assert_allclose(got, expected, atol=1e-4)
        # A single text works as well as a batch: the batch axis is dynamic.
        one = session.run(None, {"input_ids": ids[:1].numpy()})[0]
        np.testing.assert_allclose(one, expected[:1], atol=1e-4)

    def test_what_is_saved_loads_back_by_itself(self, family, bare, tmp_path):
        make_tiny(tmp_path / "files", family, bare=bare)
        folder = tmp_path / "files"
        spec, ids = text_ids(folder, ["good film", "awful plot"])
        net = hf.build_text(str(folder), CLASSES, spec.pad_id).eval()
        hf.save(net, str(tmp_path / "saved"))
        shutil.rmtree(folder)
        again = hf.load_text(str(tmp_path / "saved"), spec.pad_id).eval()
        assert (again(ids) == net(ids)).all()


class TestStrictLoading:
    """The failure this guards against is silent: transformers re-initialises any weight it cannot find and warns."""

    def make(self, tmp_path: Path, *, bare: bool, drop: str) -> Path:
        from safetensors.torch import load_file, save_file

        make_tiny(tmp_path, "bert", bare=bare)
        tensors = load_file(tmp_path / "model.safetensors")
        victim = next(k for k in tensors if drop in k)
        kept = {k: v for k, v in tensors.items() if k != victim}
        save_file(kept, tmp_path / "model.safetensors", metadata={"format": "pt"})
        return tmp_path

    @pytest.mark.parametrize("bare", [False, True], ids=["classifier", "bare backbone"])
    def test_a_missing_backbone_tensor_is_refused_not_re_initialised(self, bare, tmp_path):
        folder = self.make(tmp_path, bare=bare, drop="encoder.layer.1.output.dense.weight")
        with pytest.raises(ValueError, match="no weights for"):
            hf.build_text(str(folder), CLASSES, 0)

    def test_validation_reports_it_as_a_model_problem(self, tmp_path):
        folder = self.make(tmp_path, bare=True, drop="embeddings.word_embeddings.weight")
        with pytest.raises(ConfigError, match="cannot be loaded as a text classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_trained_model_missing_its_head_is_refused(self, tmp_path):
        folder = self.make(tmp_path, bare=False, drop="classifier.weight")
        with pytest.raises(ValueError, match="incomplete"):
            hf.load_text(str(folder), 0)

    def test_a_checkpoint_published_for_masked_language_modelling_has_no_pooler_and_still_loads(self, tmp_path):
        import torch
        from safetensors.torch import load_file, save_file

        make_tiny(tmp_path, "bert", bare=True)
        tensors = load_file(tmp_path / "model.safetensors")
        assert any("pooler" in k for k in tensors)
        kept = {k: v for k, v in tensors.items() if "pooler" not in k}
        save_file(kept, tmp_path / "model.safetensors", metadata={"format": "pt"})
        net = hf.build_text(str(tmp_path), CLASSES, 0)
        hf.freeze_backbone(net)
        assert all("classifier" in n or "pooler" in n for n, p in net.named_parameters() if p.requires_grad)
        assert isinstance(net(torch.zeros(1, 4, dtype=torch.long)), torch.Tensor)


class TestTokenizer:
    def test_a_text_is_cut_and_padded_to_the_length_with_the_special_tokens_inside_it(self, tiny):
        import torch

        spec, ids = text_ids(tiny, ["good film", "good film " * 20, ""], max_tokens=8)
        assert ids.shape == (3, 8) and ids.dtype == torch.int64
        cls, sep = 2, 3
        assert ids[0].tolist() == [cls, 4, 10, sep, 0, 0, 0, 0]
        assert ids[1][0].item() == cls and (ids[1] != 0).all() and ids[1][-1].item() == sep
        assert ids[2].tolist() == [cls, sep, 0, 0, 0, 0, 0, 0]
        assert (spec.pad_id, spec.pad_token) == (0, "[PAD]")

    def test_the_spec_round_trips_through_names_json(self, tiny):
        spec, _ = text_ids(tiny, ["good"])
        assert HFTextSpec.from_meta(json.loads(json.dumps(spec.to_meta()))) == spec

    def test_a_model_with_only_a_vocabulary_file_gets_a_fast_tokenizer_converted_from_it(self, tmp_path):
        make_tiny(tmp_path, "bert")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (tmp_path / name).unlink(missing_ok=True)
        (tmp_path / "vocab.txt").write_text("\n".join([*BERT_SPECIALS, *WORDS]), encoding="utf-8")
        tokenizer_json, pad_id, pad_token = hf.read_tokenizer(str(tmp_path))
        assert (pad_id, pad_token) == (0, "[PAD]") and json.loads(tokenizer_json)["model"]["type"] == "WordPiece"


def write_corpus(rng) -> pd.DataFrame:
    positive, negative = ["good", "great", "fine"], ["bad", "awful", "poor"]
    rows = []
    for split, n in (("train", 20), ("validation", 8), ("test", 8)):
        for i in range(n):
            label = CLASSES[i % 2]
            pool = positive if label == "positive" else negative
            w = [pool[int(j)] for j in rng.integers(0, 3, 3)]
            rows.append(
                {"text": f"the {w[0]} film and {w[1]} plot is {w[2]}", "class": label, "split": split,
                 "_theseus_item_id": f"{split}-{i}"}
            )  # fmt: skip
    return pd.DataFrame(rows)


def train_context(tmp_path: Path, config: dict, reports: list | None = None) -> TrainContext:
    return TrainContext(
        run_id="hf-text", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=(lambda epoch, split, metrics: reports.append((epoch, split, metrics))) if reports is not None
        else (lambda *a: None),
    )  # fmt: skip


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Fine-tunes the tiny BERT on a two-class toy corpus, then deletes the files it started from."""
    tmp_path = tmp_path_factory.mktemp("custom_text_run")
    df = write_corpus(np.random.default_rng(0))
    uploaded = tmp_path / "uploaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    mp.setattr(train_mod.pd, "read_parquet", lambda uri: df)

    original = tmp_path / "original"
    shutil.copytree(tiny, original)
    hyperparameters = PytorchHyperparameters(
        model_id="custom:u1", epochs=12, batch_size=8, learning_rate=0.003, max_tokens=16, seed=1
    )
    config = compile_pytorch_config(TASK, SNAPSHOT, hyperparameters, custom=ref_for(original))
    reports: list = []
    model = PytorchBackend.train(train_context(tmp_path, config, reports))
    shutil.rmtree(original)
    yield types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, tmp_path=tmp_path)
    mp.undo()


def test_training_saves_a_self_contained_hugging_face_folder(trained):
    assert isinstance(trained.model, HFTextModel)
    assert {"config.json", "model.safetensors", "names.json"} <= {p.name for p in trained.uploaded.iterdir()}
    assert not (trained.uploaded / "model.pt").exists() and not (trained.tmp_path / "work").exists()
    meta = json.loads((trained.uploaded / "names.json").read_text())
    assert (meta["hf"], meta["inputKind"], meta["names"], meta["model"]) == (True, "text", CLASSES, "custom:u1")
    assert meta["hfTokenizer"]["maxTokens"] == 16 and meta["hfTokenizer"]["padId"] == 0
    assert json.loads(meta["hfTokenizer"]["tokenizer"])["model"]["type"] == "WordLevel"
    assert trained.model.output.labels == CLASSES and trained.model.spec.max_tokens == 16


def test_training_reports_each_epoch_and_learns(trained):
    epochs_run = [e for e, s, _ in trained.reports if s == "validation"]
    assert epochs_run == list(range(1, len(epochs_run) + 1)) and 1 <= len(epochs_run) <= 12
    assert all("loss" in m and "accuracy" in m for _, _, m in trained.reports)
    assert max(m["accuracy"] for _, s, m in trained.reports if s == "validation") > 0.8


def test_a_trained_model_loads_with_the_original_files_gone(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    assert isinstance(loaded, PytorchLoadedModel) and loaded.output.labels == CLASSES


def test_predict_and_evaluate_from_the_uploaded_files(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    frame = loaded.predict(pd.DataFrame({"text": ["the great film and fine plot is good", "bad awful poor", None]}))
    probs = np.array(frame["class_probabilities"].tolist())
    assert probs.shape == (3, 2) and probs.sum(axis=1) == pytest.approx(1.0, abs=1e-4)
    assert frame["class_predictions"].tolist()[:2] == ["positive", "negative"]
    result = PytorchBackend.evaluate(loaded, trained.df, "split", "_theseus_item_id")
    assert result is not None and result.report["rowCount"] == 8 and result.report["idx2str"] == CLASSES


def test_a_text_longer_than_the_models_positions_is_refused_before_training(tiny, tmp_path, monkeypatch):
    df = write_corpus(np.random.default_rng(0))
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    hyperparameters = PytorchHyperparameters(model_id="custom:u1", max_tokens=200, epochs=1)
    config = compile_pytorch_config(TASK, SNAPSHOT, hyperparameters, custom=ref_for(tiny))
    with pytest.raises(ConfigError, match="at most 64 tokens"):
        PytorchBackend.train(train_context(tmp_path, config))


class TestManifest:
    def test_the_tokenizer_travels_whole_in_the_manifest(self, trained):
        meta = json.loads((trained.uploaded / "names.json").read_text())
        manifest = build_preprocessing_manifest("hf-text", {}, meta)
        [i] = manifest["inputs"]
        assert (i["name"], i["type"], i["column"], i["layout"]) == (
            "input_ids",
            "text",
            "text",
            "[batch, maxTokens] int64",
        )
        recipe = i["hfTokenizer"]
        assert recipe["recipe"] == "hf_tokenizer_v1"
        assert (recipe["maxTokens"], recipe["padId"], recipe["padToken"]) == (16, 0, "[PAD]")
        assert recipe["tokenizer"] == meta["hfTokenizer"]["tokenizer"]
        assert manifest["outputs"] == [{"name": "class", "type": "category", "column": "class", "classes": CLASSES}]

    def test_the_class_order_comes_from_the_trained_model(self, trained):
        meta = {**json.loads((trained.uploaded / "names.json").read_text()), "names": ["positive", "negative"]}
        assert build_preprocessing_manifest("r", {}, meta)["outputs"][0]["classes"] == ["positive", "negative"]


def test_the_exported_graph_agrees_with_the_platform_on_the_tokenizers_ids(trained, tmp_path):
    """What a client has to do is the manifest's recipe: run the stored tokenizer, feed the ids, softmax the logits."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    loaded = PytorchBackend.load(str(trained.uploaded))
    path = get_backend("pytorch").convert(loaded, "onnx", str(tmp_path / "export"))
    meta = json.loads((trained.uploaded / "names.json").read_text())
    recipe = build_preprocessing_manifest("hf-text", {}, meta)["inputs"][0]["hfTokenizer"]
    tokenizer = Tokenizer.from_str(recipe["tokenizer"])
    tokenizer.enable_truncation(recipe["maxTokens"])
    tokenizer.enable_padding(length=recipe["maxTokens"], pad_id=recipe["padId"], pad_token=recipe["padToken"])
    session = ort.InferenceSession(path)
    for sentence in ("the good film and great plot is fine", "awful poor bad", ""):
        ids = np.array([tokenizer.encode(sentence).ids], dtype=np.int64)
        logits = session.run(None, {"input_ids": ids})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        platform = loaded.predict(pd.DataFrame({"text": [sentence]}))["class_probabilities"].iloc[0]
        assert probs == pytest.approx(np.array(platform), abs=2e-3)


def test_freezing_the_backbone_trains_only_the_head_and_leaves_the_rest_untouched(tiny, tmp_path, monkeypatch):
    import torch

    df = write_corpus(np.random.default_rng(1))
    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    hyperparameters = PytorchHyperparameters(
        model_id="custom:u1", epochs=2, batch_size=8, freeze_backbone=True, max_tokens=16, seed=2
    )
    config = compile_pytorch_config(TASK, SNAPSHOT, hyperparameters, custom=ref_for(tiny))
    PytorchBackend.train(train_context(tmp_path, config))

    from safetensors.torch import load_file

    before = load_file(tiny / "model.safetensors")
    after = load_file(uploaded / "model.safetensors")
    body = [k for k in after if not k.startswith("classifier.") and "pooler" not in k]
    assert body and all(torch.equal(before[k], after[k]) for k in body)


class TestExportBundle:
    """What each generated bundle ships for a fine-tuned Hugging Face text model."""

    FORMATS = ("python_devkit", "typescript_devkit", "csharp_devkit", "java_devkit", "pwa_app", "flutter_app")

    @staticmethod
    def files(trained, tmp_path, fmt: str, sentence: str = "the good film and great plot is fine") -> dict:
        from omnitrain.plugins.export.bundle import GoldenSample, assemble_files, to_bundle_local

        loaded = PytorchBackend.load(str(trained.uploaded))
        onnx_path = get_backend("pytorch").convert(loaded, "onnx", str(tmp_path / f"export-{fmt}"))
        meta = json.loads((trained.uploaded / "names.json").read_text())
        manifest = build_preprocessing_manifest("hf-text", {}, meta)
        predictions = loaded.golden_prediction(loaded.predict(pd.DataFrame({"text": [sentence]})))
        expected = {"outputColumn": "class", "outputType": "category", "predictions": predictions}
        golden = GoldenSample(to_bundle_local(expected, "input.txt"), "input.txt", sentence.encode("utf-8"))
        made = assemble_files(
            run_name="r", task_label="Text Classification", format_id=fmt, model_bytes=Path(onnx_path).read_bytes(),
            preprocessing=manifest, golden=golden,
        )  # fmt: skip
        return {f.path: f.data for f in made}

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_the_tokenizer_ships_as_its_own_file_and_the_manifest_points_at_it(self, trained, tmp_path, fmt):
        files = self.files(trained, tmp_path, fmt)
        prefix = "assets/" if fmt == "flutter_app" else ""
        meta = json.loads((trained.uploaded / "names.json").read_text())
        assert files[f"{prefix}tokenizer.json"].decode() == meta["hfTokenizer"]["tokenizer"]
        manifest = json.loads(files[f"{prefix}preprocessing.json"])
        recipe = manifest["inputs"][0]["hfTokenizer"]
        assert "tokenizer" not in recipe and recipe["tokenizerFile"] == "tokenizer.json"
        assert (recipe["maxTokens"], recipe["padId"], recipe["padToken"]) == (16, 0, "[PAD]")

    def test_the_other_clients_say_that_they_do_not_implement_it_instead_of_pretending(self, trained, tmp_path):
        for fmt in self.FORMATS[1:]:
            readme = self.files(trained, tmp_path, fmt)["README.md"].decode()
            assert "fine-tuned Hugging Face transformer" in readme and "does not implement" in readme, fmt
            assert "the model takes word ids" not in readme, fmt

    def test_the_python_bundle_verifies_a_real_sample_end_to_end(self, trained, tmp_path):
        import subprocess
        import sys

        files = self.files(trained, tmp_path, "python_devkit")
        readme = files["README.md"].decode()
        assert "pip install onnxruntime numpy tokenizers" in readme and "tokenizer.json" in readme
        out = tmp_path / "bundle"
        for path, data in files.items():
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        ran = subprocess.run(  # noqa: S603 - our own generated script, in a temp folder
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 0 and ran.stdout.startswith("OK"), ran.stdout + ran.stderr

        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "example.py", "awful poor bad"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 0 and ran.stdout.splitlines()[0].startswith("negative"), ran.stdout + ran.stderr

        # The same bundle with a changed prediction fails.
        expected = json.loads((out / "expected.json").read_text())
        top = max(expected["predictions"], key=expected["predictions"].get)
        flipped = {label: 0.0 if label == top else 1.0 for label in expected["predictions"]}
        (out / "expected.json").write_text(json.dumps({**expected, "predictions": flipped}))
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 1 and "MISMATCH" in ran.stdout

    def test_a_client_given_the_tokenizer_inline_works_too(self, trained, tmp_path):
        """The manifest straight from the backend (tokenizer inside) is what the platform's own tests use."""
        client_source = CLIENT_PATH.read_text(encoding="utf-8")
        module = types.ModuleType("omnitrain_client_hf_inline")
        module.__file__ = str(CLIENT_PATH)
        exec(compile(client_source, str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102 - our shipped template
        loaded = PytorchBackend.load(str(trained.uploaded))
        onnx_path = get_backend("pytorch").convert(loaded, "onnx", str(tmp_path / "export"))
        meta = json.loads((trained.uploaded / "names.json").read_text())
        (tmp_path / "preprocessing.json").write_text(json.dumps(build_preprocessing_manifest("r", {}, meta)))
        client = module.OmniTrainClient(onnx_path, tmp_path / "preprocessing.json")
        for sentence in ("good great fine", "the plot is awful", ""):
            from_client = client.predict(sentence, top_k=2)
            platform = loaded.golden_prediction(loaded.predict(pd.DataFrame({"text": [sentence]})))
            assert from_client.keys() == platform.keys()
            for label, p in from_client.items():
                assert p == pytest.approx(platform[label], abs=2e-3)
