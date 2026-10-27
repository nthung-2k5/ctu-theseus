"""Bring-your-own Hugging Face token classifiers on the PyTorch backend: the kind, file checks, compile, strict loading,
how a word finds the token that speaks for it, and a real tiny train / load / predict / evaluate / ONNX / Python client
round trip. Every model is a randomly initialised few-layer network with a tiny tokenizer TRAINED here, so words really
are cut into pieces (a word-level tokenizer would never exercise the alignment); nothing is downloaded."""

import json
import shutil
import subprocess
import sys
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
from omnitrain.plugins.backends.pytorch import PytorchBackend, custom, hf, tagging, tasks  # noqa: E402
from omnitrain.plugins.backends.pytorch import train as train_mod  # noqa: E402
from omnitrain.plugins.backends.pytorch.compile import PytorchHyperparameters, compile_pytorch_config  # noqa: E402
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest  # noqa: E402
from omnitrain.plugins.backends.pytorch.model import HFTaggerModel, PytorchLoadedModel  # noqa: E402
from omnitrain.plugins.backends.pytorch.spec import HFTextSpec  # noqa: E402
from omnitrain.plugins.backends.registry import get_backend  # noqa: E402

TASK = get_task_descriptor("token_classification")
TEXT_TASK = get_task_descriptor("text_classification")
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="text", kind="inline_text"),
        ColumnSpec(name="tags", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=[],
)
TAGS = ["O", "B-LOC", "B-PER", "I-PER"]
CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"

BERT_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
ROBERTA_SPECIALS = ("<s>", "<pad>", "</s>", "<unk>")
_BASE = {"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64}
# family -> (config class, extra config arguments, bare class, tagger class, byte-level BPE tokenizer?)
FAMILIES = {
    "bert": ("BertConfig", {**_BASE, "max_position_embeddings": 64}, "BertModel", "BertForTokenClassification", False),
    "distilbert": (
        "DistilBertConfig",
        {"dim": 32, "n_layers": 2, "n_heads": 2, "hidden_dim": 64, "max_position_embeddings": 64},
        "DistilBertModel", "DistilBertForTokenClassification", False,
    ),
    "roberta": (
        "RobertaConfig",
        {**_BASE, "max_position_embeddings": 66, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2},
        "RobertaModel", "RobertaForTokenClassification", True,
    ),
    "xlm-roberta": (
        "XLMRobertaConfig",
        {**_BASE, "max_position_embeddings": 66, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2},
        "XLMRobertaModel", "XLMRobertaForTokenClassification", True,
    ),
    "albert": (
        "AlbertConfig", {**_BASE, "embedding_size": 16, "max_position_embeddings": 64},
        "AlbertModel", "AlbertForTokenClassification", False,
    ),
    "electra": (
        "ElectraConfig", {**_BASE, "embedding_size": 16, "max_position_embeddings": 64},
        "ElectraModel", "ElectraForTokenClassification", False,
    ),
}  # fmt: skip

PEOPLE = ["Anna", "Binh", "Chau", "Dung", "Emma", "Farid"]
CITIES = ["Hue", "Hanoi", "Paris", "Lyon", "Oslo", "Rome"]
RARE_PEOPLE = ["Zorblax", "Quenby", "Vortigern", "Ulrika"]  # never seen in training, cut into pieces by the tokenizer


def sentence(rng, person, city):
    """Three templates; the tag is given by the words around the name, so an unseen name is still found."""
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
    return pd.DataFrame(rows)


def write_tokenizer(path: Path, bpe: bool) -> int:
    """A tokenizer trained on the toy corpus with a vocabulary too small to hold every word, so a name is cut into
    pieces, saved the way a Hub repo carries one. WordPiece with BERT's normaliser and punctuation splitting, or the
    byte-level BPE RoBERTa uses (a word's token carries its leading space). Returns the vocabulary size."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [sentence(np.random.default_rng(i), PEOPLE[i % 6], CITIES[i % 6])[0] for i in range(60)]
    # Both alphabets in a non-initial position, so that every letter has its "##" piece and a word the corpus never saw
    # (a rare name) is cut into letters instead of becoming one unknown token.
    corpus += ["abcdefghijklmnopqrstuvwxyz", "zyxwvutsrqponmlkjihgfedcba"]
    if bpe:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=150, special_tokens=list(ROBERTA_SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        )  # fmt: skip
        tokenizer.train_from_iterator(corpus, trainer)
        tokenizer.post_processor = processors.RobertaProcessing(
            ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
            trim_offsets=True, add_prefix_space=False,
        )  # fmt: skip
        pad, unk = "<pad>", "<unk>"
    else:
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(
            vocab_size=120, special_tokens=list(BERT_SPECIALS),
            initial_alphabet=list("abcdefghijklmnopqrstuvwxyz."), show_progress=False,
        )  # fmt: skip
        tokenizer.train_from_iterator(corpus, trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
        )
        pad, unk = "[PAD]", "[UNK]"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token=pad, unk_token=unk).save_pretrained(path)
    return tokenizer.get_vocab_size()


def make_tiny(path: Path, family: str = "bert", *, bare: bool = False):
    """A random few-layer checkpoint of `family` with its tokenizer, as a tagger (5 labels) or a bare backbone."""
    import transformers

    config_name, arguments, backbone_name, tagger_name, bpe = FAMILIES[family]
    path.mkdir(parents=True, exist_ok=True)
    vocab_size = write_tokenizer(path, bpe)
    config = getattr(transformers, config_name)(vocab_size=vocab_size, **arguments)
    if bare:
        model = getattr(transformers, backbone_name)(config)
    else:
        config.num_labels = 5
        model = getattr(transformers, tagger_name)(config)
    model.save_pretrained(path, safe_serialization=True)
    return model


def ref_for(path: Path, kind: str = custom.HF_TOKEN) -> CustomModelRef:
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
    path = tmp_path_factory.mktemp("bert_tagger")
    make_tiny(path)
    return path


def read_spec(path: Path, max_tokens: int = 24) -> HFTextSpec:
    tokenizer_json, pad_id, pad_token = hf.read_tokenizer(str(path))
    return HFTextSpec(max_tokens=max_tokens, pad_id=pad_id, pad_token=pad_token, tokenizer=tokenizer_json)


def tagger_meta(spec: HFTextSpec) -> dict:
    return {
        "task": "tag", "outputName": "tags", "names": TAGS, "textColumn": "text", "hf": True,
        **{**spec.to_meta(), "inputKind": "words"},
    }  # fmt: skip


def load_client():
    module = types.ModuleType("omnitrain_client_hf_tags")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    return module


class TestWordPositions:
    """Which token speaks for a word: the first one whose span holds one of the word's characters."""

    def test_each_word_is_the_first_token_that_covers_it(self):
        # "Hue is far" joined as-is; [CLS] and [SEP] have the empty span.
        offsets = [(0, 0), (0, 3), (4, 6), (7, 10), (0, 0)]
        assert hf.word_positions(offsets, ["Hue", "is", "far"]) == [1, 2, 3]

    def test_a_word_cut_into_pieces_is_spoken_for_by_its_first_piece(self):
        offsets = [(0, 0), (0, 2), (2, 5), (5, 7), (8, 12), (0, 0)]  # "Zorblax lives": zo|rbl|ax and lives
        assert hf.word_positions(offsets, ["Zorblax", "lives"]) == [1, 4]

    def test_punctuation_split_off_a_word_belongs_to_that_word(self):
        offsets = [(0, 0), (0, 2), (2, 3), (4, 8), (0, 0)]  # "Mr. Anna": mr | . | anna
        assert hf.word_positions(offsets, ["Mr.", "Anna"]) == [1, 3]

    def test_a_token_that_carries_its_leading_space_is_found_by_its_first_character_of_the_word(self):
        offsets = [(0, 0), (0, 2), (2, 7), (0, 0)]  # a byte-level tokenizer: "Mr" then " Anna" with the space
        assert hf.word_positions(offsets, ["Mr", "Anna"]) == [1, 2]

    def test_a_word_no_token_covers_is_minus_one(self):
        # The text was cut after "is": "far" has no token; so does a word the normaliser emptied.
        assert hf.word_positions([(0, 0), (0, 3), (4, 6), (0, 0)], ["Hue", "is", "far"]) == [1, 2, -1]
        assert hf.word_positions([(0, 0), (0, 3), (6, 8), (0, 0)], ["Hue", "\x07", "is"]) == [1, -1, 2]

    def test_nothing_is_found_in_an_empty_text(self):
        assert hf.word_positions([(0, 0), (0, 0)], []) == []

    def test_labels_go_to_those_positions_only(self):
        index = {"O": 0, "B-PER": 1}
        out = tagging.token_labels([[1, 3, -1]], [["O", "B-PER", "O"]], index, 6)
        assert out.tolist() == [[-100, 0, -100, 1, -100, -100]]


class TestEncodeWords:
    def test_whitespace_between_words_does_not_change_the_tokens(self, tiny):
        tokenizer = hf.make_tokenizer(read_spec(tiny))
        ids, positions = hf.encode_words(tokenizer, ["Mr.  Anna\tlives\n in Hue .", "Mr. Anna lives in Hue ."])
        assert ids[0].tolist() == ids[1].tolist() and positions[0] == positions[1]

    def test_every_word_is_found_at_a_token_that_starts_it(self, tiny):
        import torch

        tokenizer = hf.make_tokenizer(read_spec(tiny, max_tokens=48))
        text = "Mr. Zorblax lives in Hanoi ."
        ids, [positions] = hf.encode_words(tokenizer, [text])
        assert ids.shape == (1, 48) and ids.dtype == torch.int64
        assert all(p >= 0 for p in positions) and positions == sorted(positions)
        pieces = [tokenizer.id_to_token(int(ids[0, p])) for p in positions]
        for word, piece in zip(text.split(), pieces, strict=True):
            assert piece != "[UNK]" and word.lower().startswith(piece), (word, piece)
        # The name really is cut: the next word does not start right after its first piece.
        assert positions[2] - positions[1] > 1

    def test_the_text_is_cut_at_the_length_and_words_after_the_cut_have_no_token(self, tiny):
        tokenizer = hf.make_tokenizer(read_spec(tiny, max_tokens=8))
        _, [positions] = hf.encode_words(tokenizer, ["Mr. Anna lives in Hue . Today Dr. Binh left Oslo"])
        read = [p for p in positions if p >= 0]
        assert len(positions) == 11 and positions[0] == 1 and 1 < len(read) < 11
        assert read == sorted(read) and max(read) < 8 and positions[-1] == -1
        # Once the text is cut every later word is unread, never a position that belongs to an earlier one.
        assert positions[len(read) :] == [-1] * (11 - len(read))

    def test_a_text_with_no_words_and_no_texts(self, tiny):
        tokenizer = hf.make_tokenizer(read_spec(tiny))
        ids, positions = hf.encode_words(tokenizer, ["", "  \t"])
        assert positions == [[], []] and ids.shape == (2, 24)
        assert hf.encode_words(tokenizer, [])[1] == []

    def test_a_character_the_normaliser_removes_leaves_its_word_without_a_token(self, tiny):
        tokenizer = hf.make_tokenizer(read_spec(tiny))
        _, [positions] = hf.encode_words(tokenizer, ["Mr. \x07 Anna"])
        assert positions[1] == -1 and positions[0] >= 0 and positions[2] >= 0

    def test_offsets_count_characters_not_bytes(self, tiny):
        tokenizer = hf.make_tokenizer(read_spec(tiny))
        _, [positions] = hf.encode_words(tokenizer, ["Đà Nẵng ở Hue ."])
        assert all(p >= 0 for p in positions)

    def test_a_byte_level_tokenizer_finds_the_same_words(self, tmp_path):
        make_tiny(tmp_path, "roberta")
        tokenizer = hf.make_tokenizer(read_spec(tmp_path, max_tokens=48))
        text = "Mr. Zorblax lives in Hanoi ."
        _, [positions] = hf.encode_words(tokenizer, [text])
        assert len(positions) == 6 and all(p >= 0 for p in positions) and positions == sorted(positions)


class TestKindAndValidation:
    def test_the_kind_is_offered_for_token_classification_only(self):
        assert [k.id for k in PytorchBackend.custom_model_kinds(TASK)] == [custom.HF_TOKEN]
        assert custom.HF_TOKEN not in [k.id for k in PytorchBackend.custom_model_kinds(TEXT_TASK)]
        assert custom.HF_TEXT not in [k.id for k in PytorchBackend.custom_model_kinds(TASK)]

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"model_type": "bert", "architectures": ["BertForTokenClassification"]}, custom.HF_TOKEN),
            ({"model_type": "xlm-roberta", "architectures": ["XLMRobertaForTokenClassification"]}, custom.HF_TOKEN),
            # A bare encoder is one thing for two tasks: it is suggested for the commoner, text classification.
            ({"model_type": "bert", "architectures": ["BertModel"]}, custom.HF_TEXT),
            ({"model_type": "gpt2", "architectures": ["GPT2ForTokenClassification"]}, None),
        ],
    )
    def test_the_architecture_says_which_checkpoints_are_taggers(self, config, expected):
        assert PytorchBackend.suggest_custom_kind(config) == expected

    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("bare", [False, True], ids=["tagger", "bare backbone"])
    def test_every_supported_family_validates(self, family, bare, tmp_path):
        make_tiny(tmp_path, family, bare=bare)
        PytorchBackend.validate_custom_model(TASK, ref_for(tmp_path))

    def test_every_listed_family_has_a_test_row(self):
        assert sorted(custom.SUPPORTED_TEXT_MODEL_TYPES) == sorted(FAMILIES)

    def test_a_text_classifier_kind_is_refused_for_tagging_and_the_reverse(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            PytorchBackend.validate_custom_model(TASK, ref_for(tiny, custom.HF_TEXT))
        with pytest.raises(ConfigError, match="cannot be used"):
            PytorchBackend.validate_custom_model(TEXT_TASK, ref_for(tiny, custom.HF_TOKEN))

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
        with pytest.raises(ConfigError, match="cannot be loaded as a token classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_weights_that_do_not_match_the_config_are_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, hidden_size=64, intermediate_size=128)
        with pytest.raises(ConfigError, match="cannot be loaded as a token classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_tokenizer_that_pads_with_another_id_is_refused(self, tiny, tmp_path):
        folder = copy_of(tiny, tmp_path)
        edit_config(folder, pad_token_id=3)
        with pytest.raises(ConfigError, match="pads with id 0 but the model expects 3"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))


class TestCompile:
    def compile(self, tiny, **selection):
        return compile_pytorch_config(
            TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1", **selection), custom=ref_for(tiny)
        )

    def test_a_custom_tagger_is_recorded_for_training_and_brings_its_own_vocabulary(self, tiny):
        c = self.compile(tiny, vocab_size=500, min_word_count=1, dropout=0.5)
        assert (c["inputKind"], c["task"], c["model"], c["pretrained"]) == ("words", "tag", "custom:u1", True)
        assert c["custom"] == {"kind": custom.HF_TOKEN, "path": str(tiny)}
        assert c["text"] == {"maxTokens": 128} and "network" not in c
        # The tag set is read off the data when training starts, not taken from a class list.
        assert c["classNames"] == []

    def test_it_is_fine_tuned_so_the_learning_rate_defaults_small(self, tiny):
        assert self.compile(tiny)["train"]["learningRate"] == tasks.HF_TEXT_LEARNING_RATE
        assert self.compile(tiny, learning_rate=0.01)["train"]["learningRate"] == 0.01

    def test_the_other_training_knobs_still_apply(self, tiny):
        c = self.compile(tiny, max_tokens=32, epochs=3, batch_size=8, freeze_backbone=True, use_class_weights=True)
        assert c["text"] == {"maxTokens": 32} and (c["train"]["epochs"], c["train"]["batchSize"]) == (3, 8)
        assert c["train"]["freezeBackbone"] is True and c["train"]["useClassWeights"] is True

    def test_training_from_scratch_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="from scratch"):
            self.compile(tiny, pretrained=False)

    def test_image_knobs_are_still_refused(self, tiny):
        with pytest.raises(ConfigError, match="imageSize"):
            self.compile(tiny, image_size=224)

    def test_the_text_kind_cannot_be_compiled_for_tagging(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_pytorch_config(
                TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1"), custom=ref_for(tiny, custom.HF_TEXT)
            )

    def test_the_built_in_tagger_still_refuses_to_freeze(self):
        with pytest.raises(ConfigError, match="trained from scratch"):
            compile_pytorch_config(TASK, SNAPSHOT, PytorchHyperparameters(freeze_backbone=True))


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


TEXTS = ["Mr. Anna lives in Hue .", "Today Dr. Zorblax left Oslo", "", "Mr. \x07 Binh lives in Rome ."]


@pytest.mark.parametrize("bare", [False, True], ids=["tagger", "bare backbone"])
@pytest.mark.parametrize("family", FAMILIES)
class TestNetwork:
    def build(self, folder: Path, family: str, bare: bool):
        make_tiny(folder, family, bare=bare)
        spec = read_spec(folder)
        return spec, hf.build_tokens(str(folder), TAGS, spec.pad_id)

    def test_the_pretrained_weights_are_really_loaded_and_the_head_is_new(self, family, bare, tmp_path):
        original = make_tiny(tmp_path, family, bare=bare)
        net = hf.build_tokens(str(tmp_path), TAGS, hf.read_tokenizer(str(tmp_path))[1])
        body = [k for k in original.state_dict() if not k.startswith(("classifier.", "pooler."))]
        assert backbone_matches(net, original) >= len(body) - 2
        assert net.model.config.num_labels == 4 and net.model.config.id2label[2] == "B-PER"
        spec = read_spec(tmp_path)
        ids, _ = hf.encode_words(hf.make_tokenizer(spec), ["Mr. Anna lives"])
        assert tuple(net.eval()(ids).shape) == (1, spec.max_tokens, 4)

    def test_freezing_trains_only_the_head(self, family, bare, tmp_path):
        _, net = self.build(tmp_path, family, bare)
        hf.freeze_backbone(net)
        trainable = [n for n, p in net.named_parameters() if p.requires_grad]
        assert trainable and all("classifier" in n or "pooler" in n for n in trainable)

    def test_the_exported_graph_and_the_python_client_agree_with_the_platform(self, family, bare, tmp_path):
        import onnxruntime as ort
        import torch

        spec, net = self.build(tmp_path / "files", family, bare)
        model = HFTaggerModel(net, tagger_meta(spec))
        assert isinstance(model, PytorchLoadedModel) and model.output.kind == "tokens"
        path = get_backend("pytorch").convert(model, "onnx", str(tmp_path / "export"))
        assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]

        session = ort.InferenceSession(path)
        assert [i.name for i in session.get_inputs()] == ["input_ids"]
        ids, positions = hf.encode_words(hf.make_tokenizer(spec), TEXTS)
        with torch.no_grad():
            expected = model.net(ids.to(next(model.net.parameters()).device)).cpu().numpy()
        got = session.run(None, {"input_ids": ids.numpy()})[0]
        assert got.shape == (len(TEXTS), spec.max_tokens, len(TAGS))
        np.testing.assert_allclose(got, expected, atol=1e-4)
        # A single text works as well as a batch: the batch axis is dynamic.
        np.testing.assert_allclose(session.run(None, {"input_ids": ids[:1].numpy()})[0], expected[:1], atol=1e-4)

        # The Python client reads the manifest alone and must tag every word as the platform does.
        manifest = build_preprocessing_manifest("r", {}, model.meta)
        recipe = manifest["inputs"][0]["hfTokenizer"]
        assert (recipe["recipe"], recipe["fallbackTag"], recipe["padId"]) == ("hf_word_tokens_v1", "O", spec.pad_id)
        assert manifest["outputs"] == [{"name": "tags", "type": "tags", "column": "tags", "classes": TAGS}]
        (tmp_path / "preprocessing.json").write_text(json.dumps(manifest))
        client = load_client().OmniTrainClient(path, tmp_path / "preprocessing.json")
        assert client._kind == "tags"  # noqa: SLF001
        platform = model.predict(pd.DataFrame({"text": TEXTS}))
        for i, value in enumerate(TEXTS):
            found = client.predict(value)
            assert [t["token"] for t in found] == value.split()
            assert [t["tag"] for t in found] == platform["tags_predictions"].iloc[i].split()
            assert [t["confidence"] for t in found] == pytest.approx(
                [round(p, 4) for p in platform["tags_word_probabilities"].iloc[i]], abs=2e-3
            )
        if not FAMILIES[family][4]:
            # BERT's normaliser removes a control character, so that word was never read: the fallback tag, no
            # confidence. (A byte-level tokenizer keeps every byte, so for RoBERTa the word is read like any other.)
            assert [(t["tag"], t["confidence"]) for t in client.predict(TEXTS[3])][1] == ("O", 0.0)

    def test_what_is_saved_loads_back_by_itself(self, family, bare, tmp_path):
        spec, net = self.build(tmp_path / "files", family, bare)
        net.eval()
        hf.save(net, str(tmp_path / "saved"))
        shutil.rmtree(tmp_path / "files")
        again = hf.load_text(str(tmp_path / "saved"), spec.pad_id, task="token").eval()
        ids, _ = hf.encode_words(hf.make_tokenizer(spec), TEXTS)
        assert (again(ids) == net(ids)).all()


class TestStrictLoading:
    """The failure this guards against is silent: transformers re-initialises any weight it cannot find and warns."""

    def make(self, tmp_path: Path, *, bare: bool, drop: str) -> Path:
        from safetensors.torch import load_file, save_file

        make_tiny(tmp_path, "bert", bare=bare)
        tensors = load_file(tmp_path / "model.safetensors")
        victim = next(k for k in tensors if drop in k)
        save_file({k: v for k, v in tensors.items() if k != victim}, tmp_path / "model.safetensors",
                  metadata={"format": "pt"})  # fmt: skip
        return tmp_path

    @pytest.mark.parametrize("bare", [False, True], ids=["tagger", "bare backbone"])
    def test_a_missing_backbone_tensor_is_refused_not_re_initialised(self, bare, tmp_path):
        folder = self.make(tmp_path, bare=bare, drop="encoder.layer.1.output.dense.weight")
        with pytest.raises(ValueError, match="no weights for"):
            hf.build_tokens(str(folder), TAGS, 0)

    def test_validation_reports_it_as_a_model_problem(self, tmp_path):
        folder = self.make(tmp_path, bare=True, drop="embeddings.word_embeddings.weight")
        with pytest.raises(ConfigError, match="cannot be loaded as a token classifier"):
            PytorchBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_trained_model_missing_its_head_is_refused(self, tmp_path):
        folder = self.make(tmp_path, bare=False, drop="classifier.weight")
        with pytest.raises(ValueError, match="incomplete"):
            hf.load_text(str(folder), 0, task="token")

    def test_a_text_classifier_checkpoint_becomes_a_tagger_with_a_new_head(self, tmp_path):
        """A sequence classifier's encoder is the same as a tagger's: it is a fine-tuning start like a bare one."""
        import transformers

        write_tokenizer(tmp_path, False)
        config = transformers.BertConfig(vocab_size=70, num_labels=3, **_BASE, max_position_embeddings=64)
        transformers.BertForSequenceClassification(config).save_pretrained(tmp_path, safe_serialization=True)
        net = hf.build_tokens(str(tmp_path), TAGS, 0)
        assert net.model.config.num_labels == 4


def train_context(tmp_path: Path, config: dict, reports: list | None = None) -> TrainContext:
    return TrainContext(
        run_id="hf-tags", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=(lambda epoch, split, metrics: reports.append((epoch, split, metrics))) if reports is not None
        else (lambda *a: None),
    )  # fmt: skip


def train_tagger(tmp_path: Path, mp, source: Path, df=None, **selection):
    df = make_corpus(np.random.default_rng(0)) if df is None else df
    uploaded = tmp_path / "uploaded"
    mp.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    mp.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    original = tmp_path / "original"
    shutil.copytree(source, original)
    hyperparameters = PytorchHyperparameters(
        **{"model_id": "custom:u1", "epochs": 30, "batch_size": 16, "learning_rate": 0.003, "max_tokens": 48,
           "seed": 1, **selection}
    )  # fmt: skip
    config = compile_pytorch_config(TASK, SNAPSHOT, hyperparameters, custom=ref_for(original))
    reports: list = []
    model = PytorchBackend.train(train_context(tmp_path, config, reports))
    shutil.rmtree(original)
    return types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, tmp_path=tmp_path)


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Fine-tunes the tiny BERT on the toy tagging corpus, then deletes the files it started from."""
    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        return train_tagger(tmp_path_factory.mktemp("custom_token_run"), mp, tiny)


def test_training_saves_a_self_contained_hugging_face_folder(trained):
    assert isinstance(trained.model, HFTaggerModel)
    assert {"config.json", "model.safetensors", "names.json"} <= {p.name for p in trained.uploaded.iterdir()}
    assert not (trained.uploaded / "model.pt").exists() and not (trained.tmp_path / "work").exists()
    meta = json.loads((trained.uploaded / "names.json").read_text())
    assert (meta["hf"], meta["task"], meta["inputKind"], meta["model"]) == (True, "tag", "words", "custom:u1")
    # The model's own tag order, read off the data; the head of the saved model has exactly these.
    assert meta["names"] == TAGS
    assert json.loads((trained.uploaded / "config.json").read_text())["id2label"] == {
        str(i): tag for i, tag in enumerate(TAGS)
    }
    assert meta["hfTokenizer"]["maxTokens"] == 48 and meta["hfTokenizer"]["padId"] == 0
    assert json.loads(meta["hfTokenizer"]["tokenizer"])["model"]["type"] == "WordPiece"
    assert trained.model.output.kind == "tokens" and trained.model.output.labels == TAGS


def test_training_reports_each_epoch_and_learns(trained):
    assert all({"loss", "accuracy"} <= set(m) for _, _, m in trained.reports)
    assert max(m["accuracy"] for _, s, m in trained.reports if s == "validation") > 0.9


def test_predict_gives_one_tag_per_word_and_finds_names_it_never_saw(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    assert isinstance(loaded, HFTaggerModel) and loaded.input_columns == ["text"]
    text = "Mr. Xanthippe lives in Paris ."
    predictions = loaded.predict(pd.DataFrame({"text": [text, "", "Dr. Anna left Lyon"]}))
    assert predictions["tags_predictions"].tolist()[0] == "O B-PER O O B-LOC O"
    assert predictions["tags_predictions"].tolist()[1] == "" and predictions["tags_probability"].tolist()[1] == 1.0
    assert predictions["tags_words"].tolist()[2] == ["Dr.", "Anna", "left", "Lyon"]
    output = loaded.to_output(predictions.iloc[[0]])
    assert [(t["token"], t["tag"]) for t in output["tokens"]][:2] == [("Mr.", "O"), ("Xanthippe", "B-PER")]
    assert loaded.golden_prediction(predictions.iloc[[0]]) == {}


def test_evaluation_counts_words_and_whole_entities(trained):
    result = PytorchBackend.evaluate(
        PytorchBackend.load(str(trained.uploaded)), trained.df, "split", "_theseus_item_id"
    )
    assert result is not None
    report = result.report
    assert (report["split"], report["outputType"], report["unit"], report["rowCount"]) == ("test", "tags", "word", 40)
    assert report["idx2str"] == TAGS
    assert report["overall"]["accuracy"] > 0.9 and report["overall"]["entityF1"] > 0.8


def test_words_past_the_reading_limit_get_the_outside_tag_and_no_confidence(trained):
    """A model trained on 48 tokens, asked about a text far longer: the words that fit are tagged, the rest do not."""
    loaded = PytorchBackend.load(str(trained.uploaded))
    text = "Mr. Anna lives in Hue . " + "word " * 30 + "end"
    predictions = loaded.predict(pd.DataFrame({"text": [text]}))
    tags = predictions["tags_predictions"].iloc[0].split()
    confidences = predictions["tags_word_probabilities"].iloc[0]
    assert len(tags) == len(confidences) == len(text.split()) == 37
    assert tags[-1] == "O" and confidences[-1] == 0.0 and confidences[0] > 0.0
    assert predictions["tags_probability"].iloc[0] == 0.0


def test_a_text_longer_than_the_models_positions_is_refused_before_training(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: make_corpus(np.random.default_rng(0)))
    hyperparameters = PytorchHyperparameters(model_id="custom:u1", max_tokens=200, epochs=1)
    config = compile_pytorch_config(TASK, SNAPSHOT, hyperparameters, custom=ref_for(tiny))
    with pytest.raises(ConfigError, match="at most 64 tokens"):
        PytorchBackend.train(train_context(tmp_path, config))


def test_a_snapshot_with_a_single_tag_is_refused(tiny, tmp_path, monkeypatch):
    df = make_corpus(np.random.default_rng(0)).assign(
        tags=lambda d: d["text"].map(lambda t: " ".join("O" * len(t.split())))
    )
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    config = compile_pytorch_config(
        TASK, SNAPSHOT, PytorchHyperparameters(model_id="custom:u1", epochs=1), custom=ref_for(tiny)
    )
    with pytest.raises(ConfigError, match="only one tag"):
        PytorchBackend.train(train_context(tmp_path, config))


def test_freezing_the_backbone_trains_only_the_head_and_leaves_the_rest_untouched(tiny, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file

    run = train_tagger(tmp_path, monkeypatch, tiny, epochs=2, freeze_backbone=True)
    before = load_file(tiny / "model.safetensors")
    after = load_file(run.uploaded / "model.safetensors")
    body = [k for k in after if not k.startswith("classifier.") and "pooler" not in k]
    assert body and all(torch.equal(before[k], after[k]) for k in body)
    # The new head has one row per tag of the data; the checkpoint's own had five.
    assert before["classifier.weight"].shape[0] == 5 and after["classifier.weight"].shape[0] == len(TAGS)


class TestManifest:
    def test_the_tokenizer_travels_whole_and_says_how_a_word_finds_its_token(self, trained):
        meta = json.loads((trained.uploaded / "names.json").read_text())
        manifest = build_preprocessing_manifest("hf-tags", {}, meta)
        [i] = manifest["inputs"]
        assert (i["name"], i["type"], i["column"], i["layout"]) == (
            "input_ids", "text", "text", "[batch, maxTokens] int64",
        )  # fmt: skip
        recipe = i["hfTokenizer"]
        assert recipe["recipe"] == "hf_word_tokens_v1" and recipe["fallbackTag"] == "O"
        assert (recipe["maxTokens"], recipe["padId"], recipe["padToken"]) == (48, 0, "[PAD]")
        assert recipe["tokenizer"] == meta["hfTokenizer"]["tokenizer"]
        assert "single spaces" in recipe["run"] and "first" in recipe["tags"]
        assert manifest["outputs"] == [{"name": "tags", "type": "tags", "column": "tags", "classes": TAGS}]

    def test_the_tag_order_comes_from_the_trained_model(self, trained):
        meta = {**json.loads((trained.uploaded / "names.json").read_text()), "names": ["B-PER", "O"]}
        manifest = build_preprocessing_manifest("r", {}, meta)
        assert manifest["outputs"][0]["classes"] == ["B-PER", "O"]
        # `O` is still the fallback when the tag set has it, whatever its position; otherwise the first tag.
        assert manifest["inputs"][0]["hfTokenizer"]["fallbackTag"] == "O"
        meta["names"] = ["B-PER", "B-LOC"]
        assert build_preprocessing_manifest("r", {}, meta)["inputs"][0]["hfTokenizer"]["fallbackTag"] == "B-PER"


class TestExportBundle:
    """What each generated bundle ships for a fine-tuned Hugging Face token classifier."""

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

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_the_tokenizer_ships_as_its_own_file_and_the_manifest_points_at_it(
        self, trained, tmp_path, monkeypatch, fmt
    ):
        files = self.files(trained, tmp_path, fmt, monkeypatch)
        prefix = "assets/" if fmt == "flutter_app" else ""
        meta = json.loads((trained.uploaded / "names.json").read_text())
        assert files[f"{prefix}tokenizer.json"].decode() == meta["hfTokenizer"]["tokenizer"]
        recipe = json.loads(files[f"{prefix}preprocessing.json"])["inputs"][0]["hfTokenizer"]
        assert "tokenizer" not in recipe and recipe["tokenizerFile"] == "tokenizer.json"
        assert recipe["recipe"] == "hf_word_tokens_v1"

    def test_the_other_clients_say_that_they_do_not_implement_it_instead_of_pretending(
        self, trained, tmp_path, monkeypatch
    ):
        for fmt in self.FORMATS[1:]:
            readme = self.files(trained, tmp_path, fmt, monkeypatch)["README.md"].decode()
            assert "fine-tuned Hugging Face transformer" in readme and "does not implement" in readme, fmt
            assert "first token that covers" in readme, fmt

    def test_the_python_bundle_verifies_a_real_sample_end_to_end(self, trained, tmp_path, monkeypatch):
        files = self.files(trained, tmp_path, "python_devkit", monkeypatch)
        expected = json.loads(files["expected.json"])
        assert expected["predictions"] == {} and expected["outputType"] == "tokens"
        assert len(expected["tags"]) == len(files["sample/input.txt"].decode().split())
        readme = files["README.md"].decode()
        assert "pip install onnxruntime numpy tokenizers" in readme and "tokenizer.json" in readme
        assert "Token classification with a fine-tuned Hugging Face transformer" in readme

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
            [sys.executable, "-B", "example.py", "Mr. Zorblax lives in Paris ."],
            cwd=out, capture_output=True, text=True, timeout=300,
        )  # fmt: skip
        lines = [line.split("\t") for line in ran.stdout.splitlines()]
        assert ran.returncode == 0 and [line[:2] for line in lines][:2] == [["Mr.", "O"], ["Zorblax", "B-PER"]], (
            ran.stdout + ran.stderr
        )

        # A wrong tag in expected.json is caught, naming the word.
        expected["tags"][0] = "I-PER" if expected["tags"][0] != "I-PER" else "O"
        (out / "expected.json").write_text(json.dumps(expected))
        ran = subprocess.run(  # noqa: S603
            [sys.executable, "-B", "verify.py"], cwd=out, capture_output=True, text=True, timeout=300
        )
        assert ran.returncode == 1 and "MISMATCH word 0" in ran.stdout
