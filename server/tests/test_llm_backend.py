"""The language-model backend: the prompt layout, text scores, compile, the bring-your-own kind and its file checks,
strict loading, LoRA training and merging, generation and evaluation. Every model is a randomly initialised two-layer
network (a Llama and a GPT-2, which differ in what the tokenizer adds and in how they name their layers) with a tiny
tokenizer TRAINED here; nothing is downloaded."""

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
pytest.importorskip("peft")

from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor  # noqa: E402
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, TrainContext  # noqa: E402
from omnitrain.plugins.backends.llm import LlmBackend, custom, data, metrics, network, prompt, tasks  # noqa: E402
from omnitrain.plugins.backends.llm import train as train_mod  # noqa: E402
from omnitrain.plugins.backends.llm.compile import LlmHyperparameters, compile_llm_config  # noqa: E402
from omnitrain.plugins.backends.llm.model import LlmLoadedModel  # noqa: E402
from omnitrain.plugins.backends.registry import get_backend, trainable_backends  # noqa: E402

TASK = get_task_descriptor("sequence_to_sequence")
QA_TASK = get_task_descriptor("question_answering")
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="source", kind="inline_text"),
        ColumnSpec(name="target", kind="inline_text"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=[],
)

# The toy language: six words and their translations. A model that has learned the mapping answers exactly.
PAIRS = {"red": "rouge", "blue": "bleu", "green": "vert", "one": "un", "two": "deux", "three": "trois"}
FAMILIES = ("llama", "gpt2")


def corpus(rng, per_split=(("train", 120), ("validation", 24), ("test", 24))) -> pd.DataFrame:
    words = list(PAIRS)
    rows = []
    for split, count in per_split:
        for i in range(count):
            word = words[int(rng.integers(len(words)))]
            rows.append({"source": word, "target": PAIRS[word], "split": split, "_theseus_item_id": f"{split}-{i}"})
    return pd.DataFrame(rows)


def write_tokenizer(path: Path, family: str) -> int:
    """A byte-level BPE tokenizer trained on the toy language, saved the way a Hub repo carries one. A Llama tokenizer
    puts a beginning-of-text token before every text and has no padding token; GPT-2's adds nothing."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    llama = family == "llama"
    specials = ["<unk>", "<s>", "</s>"] if llama else ["<unk>", "</s>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=200, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )  # fmt: skip
    texts = [prompt.format_prompt(["source"], [w], "target") + t for w, t in PAIRS.items()]
    tokenizer.train_from_iterator(texts * 4, trainer)
    if llama:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", tokenizer.token_to_id("<s>"))]
        )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="</s>", unk_token="<unk>", **({"bos_token": "<s>"} if llama else {})
    )
    fast.save_pretrained(path)
    return len(tokenizer.get_vocab())


def write_model(path: Path, family: str, seed: int = 0) -> None:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    path.mkdir(parents=True, exist_ok=True)
    vocab = write_tokenizer(path, family)
    torch.manual_seed(seed)
    if family == "llama":
        config = LlamaConfig(
            vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
            initializer_range=0.25,
        )  # fmt: skip
        LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)
    else:
        config = GPT2Config(
            vocab_size=vocab,
            n_embd=32,
            n_layer=2,
            n_head=2,
            n_positions=128,
            bos_token_id=1,
            eos_token_id=1,
            initializer_range=0.25,
        )
        GPT2LMHeadModel(config).save_pretrained(path, safe_serialization=True)


def ref_for(path: Path) -> CustomModelRef:
    return CustomModelRef(id="u1", kind=custom.HF_CAUSAL_LM, source_kind="upload", local_path=str(path))


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    return request.param


@pytest.fixture(scope="module")
def tiny(family, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp(f"tiny_{family}") / "model"
    write_model(path, family)
    return path


def hyperparameters(**selection) -> LlmHyperparameters:
    return LlmHyperparameters(
        **{"model_id": "custom:u1", "epochs": 30, "batch_size": 8, "learning_rate": 0.01, "max_length": 48,
           "max_new_tokens": 8, "lora_rank": 16, "seed": 1, "early_stop_patience": -1, "lr_schedule": "cosine",
           "grad_clip_norm": 1.0, **selection}
    )  # fmt: skip


class TestMetrics:
    def test_identical_texts_score_one_everywhere(self):
        assert set(metrics.score("Xin chào, bạn!", "xin chào bạn").values()) == {1.0}

    def test_punctuation_and_case_do_not_matter_but_diacritics_do(self):
        assert metrics.words("Hà Nội, Việt-Nam!") == ["hà", "nội", "việt", "nam"]
        assert metrics.exact_match("Hà Nội", "Ha Noi") == 0.0

    def test_a_decomposed_letter_equals_its_composed_one(self):
        assert metrics.exact_match("e\u0301", "\u00e9") == 1.0

    def test_token_f1_counts_a_repeated_word_as_often_as_both_have_it(self):
        # prediction "a a b" vs reference "a b b": overlap a:1 + b:1 = 2, precision 2/3, recall 2/3.
        assert metrics.token_f1("a a b", "a b b") == pytest.approx(2 / 3)

    def test_rouge_n_counts_shared_sequences(self):
        assert metrics.rouge_n("the cat sat", "the cat ran", 1) == pytest.approx(2 / 3)
        assert metrics.rouge_n("the cat sat", "the cat ran", 2) == pytest.approx(1 / 2)

    def test_rouge_l_follows_the_order_of_words(self):
        assert metrics.rouge_l("a b c d", "a x c d") == pytest.approx(3 / 4)
        assert metrics.rouge_l("c b a", "a b c") == pytest.approx(1 / 3)

    def test_a_one_word_answer_has_no_bigram_but_is_still_right_or_wrong(self):
        assert metrics.rouge_n("alpha", "alpha", 2) == 1.0
        assert metrics.rouge_n("alpha", "beta", 2) == 0.0
        assert metrics.rouge_n("alpha beta", "alpha", 2) == 0.0  # one side has a bigram, the other none

    def test_nothing_in_common_scores_zero_and_two_empty_texts_agree(self):
        assert set(metrics.score("alpha", "beta").values()) == {0.0}
        assert set(metrics.score("", "...").values()) == {1.0}
        assert metrics.token_f1("", "beta") == 0.0


class TestPrompt:
    def test_each_field_gets_a_heading_and_the_answer_heading_is_last(self):
        text = prompt.format_prompt(["context", "question"], ["It rains.", " Why wet? "], "answer")
        assert text == "### context:\nIt rains.\n\n### question:\nWhy wet?\n\n### answer:\n"

    def test_a_missing_value_is_an_empty_section(self):
        assert prompt.format_prompt(["a", "b"], [None, float("nan")], "c") == "### a:\n\n\n### b:\n\n\n### c:\n"

    def test_the_number_of_values_must_match(self):
        with pytest.raises(ValueError):
            prompt.format_prompt(["a"], [], "c")


class TestData:
    def test_prepare_drops_rows_with_no_answer_or_no_question(self):
        df = pd.DataFrame(
            {
                "source": ["a", "b", "", "d", "e"],
                "target": ["x", "", "z", None, "w"],
                "split": ["train", "train", "train", "validation", "bogus"],
            }
        )
        splits = data.prepare(df, ["source"], "target", "split")
        assert splits["train"].answers == ["x"] and splits["validation"].answers == []
        assert splits["train"].prompts == ["### source:\na\n\n### target:\n"]

    def test_a_long_prompt_loses_its_start_but_keeps_the_beginning_of_text_token(self):
        assert data.clip_prompt([1, 5, 6, 7, 8, 9], 4, bos_id=1) == [1, 7, 8, 9]
        assert data.clip_prompt([5, 6, 7, 8, 9], 3, bos_id=1) == [7, 8, 9]
        assert data.clip_prompt([1, 5, 6], 4, bos_id=1) == [1, 5, 6]

    def test_the_loss_is_on_the_answer_and_the_end_of_text_token_only(self, tiny):
        tokenizer = network.load_tokenizer(str(tiny))
        example = data.encode_example(
            tokenizer, "### source:\nred\n\n### target:\n", "rouge", max_length=48, max_new_tokens=8
        )
        answer = list(tokenizer("rouge", add_special_tokens=False)["input_ids"]) + [tokenizer.eos_token_id]
        n = len(answer)
        assert example.ids[-n:] == answer and example.labels[-n:] == answer
        assert set(example.labels[:-n]) == {data.IGNORE} and len(example.ids) == len(example.labels)

    def test_a_long_answer_is_cut_and_the_prompt_gives_way_for_it(self, tiny):
        tokenizer = network.load_tokenizer(str(tiny))
        example = data.encode_example(tokenizer, "q " * 100, "word " * 100, max_length=20, max_new_tokens=6)
        assert len(example.ids) == 20
        assert example.ids[-1] == tokenizer.eos_token_id
        assert sum(label != data.IGNORE for label in example.labels) == 6  # five answer tokens and the end

    def test_a_prompt_is_cut_the_same_way_whatever_the_length_of_its_answer(self, tiny):
        # Serving cuts to `prompt_room` because it cannot know the answer's length; training must have seen the same.
        tokenizer = network.load_tokenizer(str(tiny))
        text = prompt.format_prompt(["source"], ["word " * 60], "target")
        room = data.prompt_room(48, 8)
        served = data.encode_prompt(tokenizer, text, room)
        assert len(served) == room
        for answer in ("a", "a b c d e f g h i j k l"):
            example = data.encode_example(tokenizer, text, answer, max_length=48, max_new_tokens=8)
            assert example.ids[: len(served)] == served and len(example.ids) <= 48

    def test_collate_pads_right_shifts_the_targets_and_masks_the_padding(self):
        collate = data.collate(pad_id=9)
        packed, targets = collate([data.Example([1, 2, 3], [-100, 2, 3]), data.Example([1, 2], [-100, 2])])
        assert packed.tolist() == [[[1, 2, 3], [1, 1, 1]], [[1, 2, 9], [1, 1, 0]]]
        assert targets.tolist() == [[2, 3], [2, -100]]

    def test_prompts_are_padded_on_the_left_for_generation(self):
        ids, mask = data.left_pad([[1, 2, 3], [4]], pad_id=0)
        assert ids.tolist() == [[1, 2, 3], [0, 0, 4]] and mask.tolist() == [[1, 1, 1], [0, 0, 1]]


class TestCompile:
    def test_a_run_needs_a_bring_your_own_model(self):
        with pytest.raises(ConfigError, match="no built-in models"):
            compile_llm_config(TASK, SNAPSHOT, LlmHyperparameters(), None)
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_llm_config(TASK, SNAPSHOT, LlmHyperparameters(model_id="custom:gone"), None)

    def test_only_the_causal_lm_kind_and_the_text_tasks_are_accepted(self, tmp_path):
        wrong = CustomModelRef(id="u1", kind="hf_image", source_kind="upload", local_path=str(tmp_path))
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_llm_config(TASK, SNAPSHOT, hyperparameters(), wrong)
        with pytest.raises(ConfigError, match="does not support"):
            compile_llm_config(
                get_task_descriptor("image_classification"), SNAPSHOT, hyperparameters(), ref_for(tmp_path)
            )

    def test_the_config_names_the_columns_and_the_defaults(self, tmp_path):
        config = compile_llm_config(TASK, SNAPSHOT, LlmHyperparameters(model_id="custom:u1"), ref_for(tmp_path))
        assert config["inputColumns"] == ["source"] and config["outputColumn"] == "target"
        assert config["custom"] == {"kind": "hf_causal_lm", "path": str(tmp_path)}
        assert config["text"] == {"maxLength": 512, "maxNewTokens": 128}
        assert config["lora"] == {"rank": 8, "alpha": 16, "dropout": 0.05}
        assert config["train"]["epochs"] == 3 and config["train"]["batchSize"] == 4 and config["train"]["patience"] == 2

    def test_a_question_answering_prompt_has_both_its_input_columns(self, tmp_path):
        snapshot = SnapshotContext(
            columns=[
                ColumnSpec(name="context", kind="inline_text"),
                ColumnSpec(name="question", kind="inline_text"),
                ColumnSpec(name="answer", kind="inline_text"),
                ColumnSpec(name="split", kind="split"),
            ]
        )
        config = compile_llm_config(QA_TASK, snapshot, hyperparameters(), ref_for(tmp_path))
        assert config["inputColumns"] == ["context", "question"] and config["outputColumn"] == "answer"

    def test_alpha_follows_the_rank_unless_it_is_given(self, tmp_path):
        config = compile_llm_config(TASK, SNAPSHOT, hyperparameters(lora_rank=32), ref_for(tmp_path))
        assert config["lora"]["alpha"] == 64
        config = compile_llm_config(TASK, SNAPSHOT, hyperparameters(lora_rank=32, lora_alpha=10), ref_for(tmp_path))
        assert config["lora"]["alpha"] == 10

    def test_an_answer_must_be_shorter_than_prompt_and_answer_together(self, tmp_path):
        with pytest.raises(ConfigError, match="must be shorter"):
            compile_llm_config(TASK, SNAPSHOT, hyperparameters(max_length=64, max_new_tokens=64), ref_for(tmp_path))

    def test_a_knob_the_backend_cannot_honour_is_refused_not_ignored(self, tmp_path):
        with pytest.raises(ConfigError, match="mixedPrecision"):
            compile_llm_config(TASK, SNAPSHOT, hyperparameters(mixed_precision=True), ref_for(tmp_path))

    def test_early_stopping_can_be_switched_off(self, tmp_path):
        config = compile_llm_config(
            TASK, SNAPSHOT, hyperparameters(early_stop_patience=-1, epochs=7), ref_for(tmp_path)
        )
        assert config["train"]["patience"] == 7

    def test_the_memory_knobs_default_to_off_and_are_carried_into_the_config(self, tmp_path):
        config = compile_llm_config(TASK, SNAPSHOT, LlmHyperparameters(model_id="custom:u1"), ref_for(tmp_path))
        assert config["quantization"] == "none"
        assert (config["train"]["gradAccumulation"], config["train"]["gradientCheckpointing"]) == (1, False)
        config = compile_llm_config(
            TASK,
            SNAPSHOT,
            hyperparameters(grad_accumulation=4, gradient_checkpointing=True, quantization="4bit"),
            ref_for(tmp_path),
        )
        assert config["quantization"] == "4bit"
        assert (config["train"]["gradAccumulation"], config["train"]["gradientCheckpointing"]) == (4, True)

    def test_an_unknown_quantization_is_rejected(self):
        with pytest.raises(ValueError):
            LlmHyperparameters(quantization="2bit")

    def test_the_form_covers_every_knob_the_backend_has(self):
        names = {s.name for s in LlmBackend.hyperparameter_specs(TASK)}
        assert {"loraRank", "loraAlpha", "loraDropout", "maxLength", "maxNewTokens", "epochs", "learningRate"} <= names
        assert {"gradAccumulation", "gradientCheckpointing", "quantization"} <= names
        assert "mixedPrecision" not in names
        accepted = set(LlmHyperparameters.model_fields)
        assert {"model_id", "lora_rank", "max_length"} <= accepted


class TestKind:
    def test_the_kind_is_offered_for_the_text_tasks_only(self):
        for task in tasks.LLM_TASKS:
            assert [k.id for k in LlmBackend.custom_model_kinds(get_task_descriptor(task))] == ["hf_causal_lm"]
        assert LlmBackend.custom_model_kinds(get_task_descriptor("text_classification")) == []

    @pytest.mark.parametrize(
        ("architectures", "expected"),
        [
            (["LlamaForCausalLM"], "hf_causal_lm"),
            (["GPT2LMHeadModel"], "hf_causal_lm"),
            (["BertModel"], None),
            ([], None),
        ],
    )
    def test_the_architecture_says_whether_it_is_a_causal_lm(self, architectures, expected):
        assert LlmBackend.suggest_custom_kind({"architectures": architectures}) == expected

    def test_a_tiny_model_is_accepted(self, tiny):
        LlmBackend.validate_custom_model(TASK, ref_for(tiny))

    def test_the_kind_must_suit_the_task(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            LlmBackend.validate_custom_model(get_task_descriptor("text_classification"), ref_for(tiny))

    def test_code_that_ships_with_a_model_is_never_run(self, tiny, tmp_path):
        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        config = json.loads((copy / "config.json").read_text())
        (copy / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "x.Y"}}))
        with pytest.raises(ConfigError, match="custom code"):
            LlmBackend.validate_custom_model(TASK, ref_for(copy))

    def test_a_model_that_is_not_a_language_model_is_refused(self, tiny, tmp_path):
        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        config = json.loads((copy / "config.json").read_text())
        (copy / "config.json").write_text(json.dumps({**config, "architectures": ["BertModel"]}))
        with pytest.raises(ConfigError, match="Not a causal language model"):
            LlmBackend.validate_custom_model(TASK, ref_for(copy))

    def test_weights_that_are_not_safetensors_are_refused(self, tiny, tmp_path):
        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        (copy / "model.safetensors").rename(copy / "pytorch_model.bin")
        with pytest.raises(ConfigError, match="safetensors"):
            LlmBackend.validate_custom_model(TASK, ref_for(copy))

    def test_a_model_without_a_tokenizer_is_refused(self, tiny, tmp_path):
        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        (copy / "tokenizer.json").unlink()
        with pytest.raises(ConfigError, match="no tokenizer"):
            LlmBackend.validate_custom_model(TASK, ref_for(copy))

    def test_a_tokenizer_with_no_end_of_text_token_is_refused(self, tiny, tmp_path):
        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        body = json.loads((copy / "tokenizer_config.json").read_text())
        body.pop("eos_token")
        (copy / "tokenizer_config.json").write_text(json.dumps(body))
        with pytest.raises(ConfigError, match="end-of-text"):
            LlmBackend.validate_custom_model(TASK, ref_for(copy))

    def test_the_position_limit_is_read_from_whichever_key_the_family_uses(self):
        assert custom.position_limit({"max_position_embeddings": 4096}) == 4096
        assert custom.position_limit({"n_positions": 1024}) == 1024
        assert custom.position_limit({}) is None


class TestRegistration:
    def test_the_backend_is_listed_and_exports_only_its_folder(self):
        backend = get_backend("llm")
        assert backend is LlmBackend and list(backend.artifacts) == ["hf_model"] and backend.models(TASK) == []
        assert backend.remote_custom_models is True and backend.metadata_filename == "names.json"
        assert backend.available() is None and backend.supports(TASK)
        assert not backend.supports(get_task_descriptor("image_classification"))

    def test_ludwig_stays_the_default_for_the_generative_tasks(self):
        for task in tasks.LLM_TASKS:
            assert [b.id for b in trainable_backends(get_task_descriptor(task))] == ["ludwig", "llm"]

    def test_listing_the_backend_does_not_import_torch(self):
        code = (
            "import sys\n"
            "from omnitrain.plugins.backends.registry import list_backends\n"
            "assert 'llm' in [b.id for b in list_backends()]\n"
            "from omnitrain.plugins.backends.llm import LlmBackend\n"
            "LlmBackend.custom_model_kinds(__import__('omnitrain.domain.task_registry', fromlist=['x'])"
            ".get_task_descriptor('summarization'))\n"
            "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]\n"
            "assert not bad, bad\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr[-800:]


class TestNetwork:
    def test_a_missing_tensor_is_an_error_not_a_random_layer(self, tiny, tmp_path):
        from safetensors.torch import load_file, save_file

        copy = tmp_path / "m"
        shutil.copytree(tiny, copy)
        tensors = load_file(copy / "model.safetensors")
        victim = next(k for k in tensors if "layers.1" in k or "h.1" in k)
        del tensors[victim]
        save_file(tensors, copy / "model.safetensors", metadata={"format": "pt"})
        import torch

        with pytest.raises(ValueError, match="incomplete"):
            network.load_base(str(copy), torch.float32)

    def test_only_the_adapters_train_and_merging_changes_no_answer(self, tiny):
        import torch

        base = network.load_base(str(tiny), torch.float32).eval()
        ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11]])
        before = base(input_ids=ids).logits
        peft_model = network.add_lora(base, rank=4, alpha=8, dropout=0.0)
        trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        assert trainable and all(network.is_adapter(n) for n in trainable)
        # Zero-initialised: the adapted model starts out exactly as the base model was.
        torch.testing.assert_close(peft_model.eval()(input_ids=ids).logits, before)
        # Move the adapters, then check that the merged model computes what the adapted one did.
        for p in peft_model.parameters():
            if p.requires_grad:
                p.data += 0.05 * torch.randn_like(p)
        adapted = peft_model.eval()(input_ids=ids).logits
        assert not torch.allclose(adapted, before, atol=1e-3)
        merged = network.merge(peft_model).eval()
        assert not any(network.is_adapter(n) for n, _ in merged.named_parameters())
        torch.testing.assert_close(merged(input_ids=ids).logits, adapted, atol=1e-4, rtol=1e-4)

    def test_the_causal_net_scores_the_next_token_and_ignores_padding(self, tiny):
        import torch

        net = network.CausalNet(network.load_base(str(tiny), torch.float32)).eval()
        short = torch.tensor([[[1, 5, 6, 7], [1, 1, 1, 1]]])
        padded = torch.tensor([[[1, 5, 6, 7, 0, 0], [1, 1, 1, 1, 0, 0]]])
        a, b = net(short), net(padded)
        assert a.shape[:2] == (1, 3) and b.shape[:2] == (1, 5)
        torch.testing.assert_close(a, b[:, :3], atol=1e-5, rtol=1e-5)


def train_context(tmp_path: Path, config: dict, reports: list | None = None, abort=lambda: None) -> TrainContext:
    return TrainContext(
        run_id="llm-run", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=abort,
        _report=(lambda epoch, split, metrics_: reports.append((epoch, split, metrics_))) if reports is not None
        else (lambda *a: None),
    )  # fmt: skip


def train_model(
    tmp_path: Path, mp, source: Path, df=None, task=TASK, snapshot=SNAPSHOT, abort=lambda: None, **selection
):
    df = corpus(np.random.default_rng(0)) if df is None else df
    uploaded = tmp_path / "uploaded"
    mp.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    mp.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    original = tmp_path / "original"
    shutil.copytree(source, original)
    config = compile_llm_config(task, snapshot, hyperparameters(**selection), ref_for(original))
    reports: list = []
    model = LlmBackend.train(train_context(tmp_path, config, reports, abort))
    shutil.rmtree(original)
    return types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, tmp_path=tmp_path)


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Fine-tunes the tiny model on the toy language, then deletes the files it started from."""
    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        return train_model(tmp_path_factory.mktemp("llm_run"), mp, tiny)


class TestTraining:
    def test_the_result_is_a_self_contained_folder_with_no_adapter_left(self, trained):
        from safetensors import safe_open

        names = {p.name for p in trained.uploaded.iterdir()}
        assert {"config.json", "model.safetensors", "names.json", "tokenizer.json"} <= names
        assert not (trained.tmp_path / "work").exists()
        with safe_open(trained.uploaded / "model.safetensors", "pt") as f:
            assert not any("lora" in key for key in f.keys())  # noqa: SIM118
        meta = json.loads((trained.uploaded / "names.json").read_text())
        assert (meta["task"], meta["inputKind"], meta["model"]) == ("sequence_to_sequence", "llm", "custom:u1")
        assert meta["inputColumns"] == ["source"] and meta["outputColumn"] == "target"
        assert meta["text"] == {"maxLength": 48, "maxNewTokens": 8}

    def test_each_epoch_reports_a_training_and_a_validation_loss_that_falls(self, trained):
        by_split = {s: [m for _, split, m in trained.reports if split == s] for s in ("train", "validation")}
        assert by_split["train"] and len(by_split["train"]) == len(by_split["validation"])
        assert all("loss" in m and "accuracy" in m for m in by_split["validation"])
        assert by_split["validation"][-1]["loss"] < by_split["validation"][0]["loss"] * 0.5

    def test_the_trained_model_writes_the_answers_it_was_taught(self, trained):
        assert isinstance(trained.model, LlmLoadedModel)
        test = trained.df[trained.df["split"] == "test"]
        predictions = trained.model.predict(test[["source"]])
        assert list(predictions["target_predictions"]) == list(test["target"])

    def test_loading_from_the_uploaded_folder_gives_the_same_answers(self, trained):
        loaded = LlmBackend.load(str(trained.uploaded))
        frame = pd.DataFrame({"source": list(PAIRS)})
        assert list(loaded.predict(frame)["target_predictions"]) == list(PAIRS.values())

    def test_a_prompt_is_answered_the_same_alone_or_in_a_padded_batch(self, trained):
        # Prompts of different lengths share a batch (left-padded): the answer must not depend on who they sit with.
        frame = pd.DataFrame({"source": ["red", "three", "one", "green"]})
        together = list(trained.model.predict(frame)["target_predictions"])
        alone = [trained.model.predict(frame.iloc[[i]])["target_predictions"].iloc[0] for i in range(len(frame))]
        assert together == alone

    def test_the_output_shapes_the_inference_route_expects(self, trained):
        model = trained.model
        predictions = model.predict(pd.DataFrame({"source": ["blue"]}))
        assert model.input_columns == ["source"] and (model.output.name, model.output.kind) == ("target", "text")
        assert model.output.labels is None
        assert model.to_output(predictions) == {"kind": "text", "feature": "target", "text": "bleu"}
        assert model.golden_prediction(predictions) == {}

    def test_the_two_families_both_reach_the_same_answers(self, family, trained):
        assert family in FAMILIES
        assert trained.model.predict(pd.DataFrame({"source": ["two"]}))["target_predictions"].iloc[0] == "deux"


def test_an_abort_stops_training_without_uploading(tiny, tmp_path, monkeypatch):
    from omnitrain.jobs.abort import TrainingAborted

    calls = {"n": 0}

    def abort():
        calls["n"] += 1
        if calls["n"] > 6:
            raise TrainingAborted

    with pytest.raises(TrainingAborted):
        train_model(tmp_path, monkeypatch, tiny, abort=abort)
    assert not (tmp_path / "uploaded").exists() and not (tmp_path / "work").exists()


def test_a_run_longer_than_the_models_positions_is_refused_before_training(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: corpus(np.random.default_rng(0)))
    config = compile_llm_config(TASK, SNAPSHOT, hyperparameters(max_length=500, epochs=1), ref_for(tiny))
    with pytest.raises(ConfigError, match="at most 128 tokens"):
        LlmBackend.train(train_context(tmp_path, config))


def test_a_split_with_no_usable_rows_is_refused(tiny, tmp_path, monkeypatch):
    df = corpus(np.random.default_rng(0)).assign(target="")
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    config = compile_llm_config(TASK, SNAPSHOT, hyperparameters(epochs=1), ref_for(tiny))
    with pytest.raises(ConfigError, match="train split has no usable rows"):
        LlmBackend.train(train_context(tmp_path, config))


def test_a_two_field_task_trains_on_both_fields(tiny, tmp_path, monkeypatch):
    """Question answering: the answer depends on the QUESTION field; the context field is the same in every row."""
    rng = np.random.default_rng(0)
    base = corpus(rng, (("train", 72), ("validation", 12), ("test", 12)))
    df = pd.DataFrame(
        {
            "context": "A glossary.",
            "question": base["source"],
            "answer": base["target"],
            "split": base["split"],
            "_theseus_item_id": base["_theseus_item_id"],
        }
    )
    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="context", kind="inline_text"),
            ColumnSpec(name="question", kind="inline_text"),
            ColumnSpec(name="answer", kind="inline_text"),
            ColumnSpec(name="split", kind="split"),
        ]
    )
    run = train_model(tmp_path, monkeypatch, tiny, df=df, task=QA_TASK, snapshot=snapshot, max_length=96, epochs=50)
    assert run.model.input_columns == ["context", "question"]
    asked = pd.DataFrame({"context": ["A glossary."] * 2, "question": ["red", "three"]})
    assert list(run.model.predict(asked)["answer_predictions"]) == ["rouge", "trois"]


class TestEvaluation:
    def test_the_report_scores_each_written_answer_against_the_model_s(self, trained):
        result = LlmBackend.evaluate(trained.model, trained.df, "split", "_theseus_item_id")
        report = result.report
        assert (report["split"], report["outputFeature"], report["outputType"]) == ("test", "target", "text")
        assert report["rowCount"] == report["totalRows"] == 24
        overall = report["overall"]
        assert overall["exactMatch"] == overall["tokenF1"] == overall["rougeL"] == 1.0
        assert 0 < overall["loss"] < 0.5 and overall["perplexity"] == pytest.approx(np.exp(overall["loss"]), abs=1e-3)

    def test_the_download_pairs_each_prediction_with_the_actual_answer_and_the_item(self, trained):
        result = LlmBackend.evaluate(trained.model, trained.df, "split", "_theseus_item_id")
        frame = result.predictions
        test = trained.df[trained.df["split"] == "test"].reset_index(drop=True)
        assert list(frame["_theseus_item_id"]) == list(test["_theseus_item_id"])
        assert list(frame["target_actual"]) == list(test["target"])
        assert list(frame["target_predictions"]) == list(test["target"])

    def test_the_headline_is_rouge_l(self, trained):
        from omnitrain.services.evaluation_score import score_from_overall

        result = LlmBackend.evaluate(trained.model, trained.df, "split", "_theseus_item_id")
        score = score_from_overall(result.report["overall"])
        assert (score.metric, score.label, score.higher_is_better) == ("rougeL", "ROUGE-L", True)

    def test_only_the_first_rows_are_generated_and_the_report_says_so(self, trained, monkeypatch):
        monkeypatch.setattr(tasks, "EVAL_MAX_ROWS", 5)
        result = LlmBackend.evaluate(trained.model, trained.df, "split", "_theseus_item_id")
        assert (result.report["rowCount"], result.report["totalRows"]) == (5, 24)
        assert len(result.predictions) == 5

    def test_a_wrong_answer_lowers_the_scores(self, trained):
        df = trained.df.copy()
        df.loc[df["split"] == "test", "target"] = "completely different words"
        overall = LlmBackend.evaluate(trained.model, df, "split", "_theseus_item_id").report["overall"]
        assert overall["exactMatch"] == 0.0 and overall["loss"] > 1.0

    def test_with_no_written_answer_there_is_nothing_to_evaluate(self, trained):
        df = trained.df.assign(target="")
        assert LlmBackend.evaluate(trained.model, df, "split", "_theseus_item_id") is None

    def test_an_unsplit_table_is_evaluated_whole(self, trained):
        df = trained.df.drop(columns=["split"])
        report = LlmBackend.evaluate(trained.model, df.head(6), "split", "_theseus_item_id").report
        assert (report["split"], report["rowCount"]) == ("full", 6)


class TestExport:
    """The `hf_model` format: the merged folder zipped by `convert`, laid out as a bundle, and the bundle's own Python
    client run in a fresh interpreter against the platform's answer for a sample."""

    @pytest.fixture(scope="class")
    @classmethod
    def bundle(cls, trained, tmp_path_factory):
        from _pytest.monkeypatch import MonkeyPatch

        from omnitrain.jobs import export as export_job
        from omnitrain.plugins.export import bundle as B
        from omnitrain.plugins.export.common import GoldenSample

        work = tmp_path_factory.mktemp("llm_export")
        archive = Path(LlmBackend.convert(trained.model, "hf_model", str(work)))
        with MonkeyPatch.context() as mp:
            mp.setattr(export_job.pd, "read_parquet", lambda path: trained.df)
            mp.setattr(export_job.storage, "s3fs_path", lambda bucket, key: key)
            golden = export_job._build_golden_sample(LlmBackend, str(trained.uploaded), "snapshot.parquet")
        filename, data_ = B.resolve_sample_file(golden["inputValue"], None)
        meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
        files = B.assemble_files(
            run_name="toy", task_label="Sequence to sequence", format_id="hf_model",
            model_bytes=archive.read_bytes(), preprocessing=LlmBackend.preprocessing_manifest("run-1", {}, meta),
            golden=GoldenSample(B.to_bundle_local(golden, filename), filename, data_),
        )  # fmt: skip
        root = work / "unpacked"
        root.mkdir()
        (work / "bundle.zip").write_bytes(B.make_zip(files))
        shutil.unpack_archive(work / "bundle.zip", root)
        return types.SimpleNamespace(archive=archive, golden=golden, files=files, root=root)

    @staticmethod
    def run_script(root: Path, *args: str):
        # Not `-I`: the scripts import the client that sits beside them.
        return subprocess.run(
            [sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=600, check=False
        )

    def test_only_this_format_is_offered_for_a_language_model_run(self):
        from omnitrain.plugins.export.registry import list_export_formats

        assert [f.id for f in list_export_formats(TASK, LlmBackend)] == ["hf_model"]
        # A backend with no folder to bundle is not offered it.
        assert "hf_model" not in [f.id for f in list_export_formats(TASK, get_backend("ludwig"))]
        assert [f.id for f in list_export_formats(get_task_descriptor("image_classification"), LlmBackend)] == []

    def test_the_artifact_is_the_saved_folder_stored_uncompressed(self, bundle, trained):
        import zipfile

        with zipfile.ZipFile(bundle.archive) as z:
            names = set(z.namelist())
            assert {"config.json", "model.safetensors", "names.json", "tokenizer.json"} <= names
            assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_STORED}
            assert z.read("names.json") == (trained.uploaded / "names.json").read_bytes()
        assert bundle.archive.name == "model.zip"

    def test_the_golden_sample_carries_the_text_the_model_wrote(self, bundle, trained):
        sample = trained.df[trained.df["split"] == "test"].iloc[0]
        assert bundle.golden["outputType"] == "text" and bundle.golden["predictions"] == {}
        assert bundle.golden["text"] == sample["target"] and bundle.golden["inputValue"] == sample["source"]
        expected = json.loads((bundle.root / "expected.json").read_text(encoding="utf-8"))
        assert expected["text"] == sample["target"] and expected["sampleFile"] == "sample/input.txt"

    def test_the_bundle_holds_the_folder_the_manifest_and_the_client(self, bundle):
        names = {p.relative_to(bundle.root).as_posix() for p in bundle.root.rglob("*") if p.is_file()}
        assert {
            "model/config.json", "model/model.safetensors", "model/names.json", "model/tokenizer.json",
            "preprocessing.json", "omnitrain_llm.py", "example.py", "verify.py", "expected.json", "sample/input.txt",
            "README.md",
        } <= names  # fmt: skip
        assert not {n for n in names if n.endswith((".onnx", ".pt2"))}
        manifest = json.loads((bundle.root / "preprocessing.json").read_text(encoding="utf-8"))
        generation = manifest["generation"]
        assert generation["recipe"] == "causal_lm_prompt_v1" and generation["inputColumns"] == ["source"]
        assert (generation["maxLength"], generation["maxNewTokens"], generation["promptRoom"]) == (48, 8, 40)
        assert manifest["outputs"] == [{"name": "target", "type": "text", "column": "target"}]
        readme = (bundle.root / "README.md").read_text(encoding="utf-8")
        assert "python verify.py" in readme and "{{" not in readme

    def test_the_shipped_client_writes_what_the_platform_wrote(self, bundle):
        done = self.run_script(bundle.root, "verify.py")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "OK" in done.stdout

    def test_the_example_answers_for_a_text_given_on_the_command_line(self, bundle):
        done = self.run_script(bundle.root, "example.py", "red")
        assert done.returncode == 0, done.stdout + done.stderr
        assert done.stdout.strip().splitlines()[-1] == "rouge"

    def test_a_different_answer_fails_the_check(self, bundle, tmp_path):
        copy = tmp_path / "bundle"
        shutil.copytree(bundle.root, copy)
        expected = json.loads((copy / "expected.json").read_text(encoding="utf-8"))
        expected["text"] = "something else"
        (copy / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
        done = self.run_script(copy, "verify.py")
        assert done.returncode == 1 and "MISMATCH" in done.stdout

    def test_a_recipe_the_client_does_not_know_is_refused(self, bundle, tmp_path):
        copy = tmp_path / "bundle"
        shutil.copytree(bundle.root, copy)
        manifest = json.loads((copy / "preprocessing.json").read_text(encoding="utf-8"))
        manifest["generation"]["recipe"] = "causal_lm_prompt_v2"
        (copy / "preprocessing.json").write_text(json.dumps(manifest), encoding="utf-8")
        done = self.run_script(copy, "example.py", "red")
        assert done.returncode != 0 and "causal_lm_prompt_v2" in done.stderr

    def test_a_model_too_big_for_a_bundle_is_refused(self, trained, tmp_path, monkeypatch):
        from omnitrain.plugins.backends.llm import artifacts

        monkeypatch.setattr(artifacts, "MAX_EXPORT_BYTES", 1000)
        with pytest.raises(ValueError, match="more than the"):
            LlmBackend.convert(trained.model, "hf_model", str(tmp_path))

    def test_no_other_artifact_is_made(self, trained, tmp_path):
        with pytest.raises(ValueError, match="onnx"):
            LlmBackend.convert(trained.model, "onnx", str(tmp_path))

    def test_the_manifest_needs_the_run_s_metadata(self):
        with pytest.raises(ValueError, match="names.json"):
            LlmBackend.preprocessing_manifest("run-1", {}, None)


class TestMemory:
    def test_an_adapter_merged_into_a_fresh_copy_equals_one_merged_in_place(self, tiny):
        """How a model trained on a 4-bit base is merged: the adapters are written down and folded into a full-precision
        reload. That route has to give the very model the in-place merge does."""
        import torch

        ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11]])
        peft_model = network.add_lora(network.load_base(str(tiny), torch.float32), rank=4, alpha=8, dropout=0.0)
        torch.manual_seed(3)
        for p in peft_model.parameters():
            if p.requires_grad:
                p.data += 0.05 * torch.randn_like(p)
        folder = network.save_adapter(peft_model)
        try:
            fresh = network.merge_saved(str(tiny), folder.name, torch.float32).eval()
        finally:
            folder.cleanup()
        in_place = network.merge(peft_model).eval()
        torch.testing.assert_close(fresh(input_ids=ids).logits, in_place(input_ids=ids).logits, atol=1e-5, rtol=1e-5)

    def test_checkpointing_changes_no_logit_and_still_passes_gradients_to_the_adapters(self, tiny):
        import torch

        def logits_and_grads(checkpointing: bool):
            torch.manual_seed(0)
            base = network.prepare(
                network.load_base(str(tiny), torch.float32), quantized=False, checkpointing=checkpointing
            )
            net = network.CausalNet(network.add_lora(base, rank=4, alpha=8, dropout=0.0)).train()
            for p in net.parameters():
                if p.requires_grad:
                    p.data += 0.05 * torch.randn_like(p)
            packed = torch.tensor([[[1, 5, 6, 7, 8, 9], [1, 1, 1, 1, 1, 1]]])
            out = net(packed)
            out.square().mean().backward()
            grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.requires_grad and p.grad is not None}
            return out.detach(), grads

        plain, plain_grads = logits_and_grads(False)
        saved, saved_grads = logits_and_grads(True)
        torch.testing.assert_close(plain, saved, atol=1e-5, rtol=1e-5)
        assert plain_grads and set(plain_grads) == set(saved_grads)
        for name in plain_grads:
            torch.testing.assert_close(plain_grads[name], saved_grads[name], atol=1e-4, rtol=1e-3)

    def test_quantizing_is_refused_where_it_cannot_work(self, tiny, tmp_path, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert network.quantization_unavailable("none") is None
        assert "CUDA" in network.quantization_unavailable("4bit")
        monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: corpus(np.random.default_rng(0)))
        config = compile_llm_config(TASK, SNAPSHOT, hyperparameters(quantization="4bit", epochs=1), ref_for(tiny))
        with pytest.raises(ConfigError, match="cannot be held in 4bit"):
            LlmBackend.train(train_context(tmp_path, config))

    def test_a_missing_package_is_the_reason_when_there_is_no_bitsandbytes(self, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name, *a: None if name == "bitsandbytes" else real(name, *a)
        )
        assert "bitsandbytes" in network.quantization_unavailable("8bit")

    def test_batches_added_up_before_an_update_train_as_well_as_one_big_batch(self, tiny, tmp_path, monkeypatch):
        run = train_model(tmp_path, monkeypatch, tiny, batch_size=4, grad_accumulation=2)
        validation = [m["loss"] for _, split, m in run.reports if split == "validation"]
        # The same number of updates as batches of 8, and the loss falls along the same curve. (Exact answers are not
        # asserted: the tiny GPT-2 is a marginal learner at this length, with or without accumulation.)
        assert validation[-1] < validation[0] * 0.25

    def test_a_model_trained_with_checkpointing_is_still_an_ordinary_model(self, tiny, tmp_path, monkeypatch):
        run = train_model(tmp_path, monkeypatch, tiny, gradient_checkpointing=True)
        test = run.df[run.df["split"] == "test"]
        assert list(run.model.predict(test[["source"]])["target_predictions"]) == list(test["target"])
        # What was saved does not tell a server to run without the cache, which would make generation needlessly slow.
        assert json.loads((run.uploaded / "config.json").read_text()).get("use_cache", True) is not False


def _gpu_with_bitsandbytes() -> bool:
    import importlib.util

    import torch

    return torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None


@pytest.mark.skipif(not _gpu_with_bitsandbytes(), reason="needs a CUDA GPU and bitsandbytes")
class TestQuantizedTraining:
    @pytest.mark.parametrize("quantization", ["4bit", "8bit"])
    def test_a_model_trained_on_a_quantized_base_is_saved_at_full_precision_and_learns(
        self, quantization, tiny, tmp_path, monkeypatch
    ):
        from safetensors import safe_open

        run = train_model(tmp_path, monkeypatch, tiny, quantization=quantization)
        validation = [m["loss"] for _, split, m in run.reports if split == "validation"]
        assert validation[-1] < validation[0] * 0.5
        # Nothing of the quantisation is left in the files: they load with no bitsandbytes, as any other model does.
        config = json.loads((run.uploaded / "config.json").read_text())
        assert "quantization_config" not in config
        with safe_open(run.uploaded / "model.safetensors", "pt") as f:
            keys = list(f.keys())  # noqa: SIM118
            assert not any("lora" in k or "absmax" in k or "quant" in k for k in keys)
            assert all(f.get_tensor(k).dtype.is_floating_point for k in keys)
        test = run.df[run.df["split"] == "test"]
        answers = list(run.model.predict(test[["source"]])["target_predictions"])
        assert np.mean([a == t for a, t in zip(answers, test["target"], strict=True)]) > 0.9


class TestAccumulation:
    @staticmethod
    def _stepper(accumulate: int, steps_per_epoch: int, **hp):
        import torch

        from omnitrain.plugins.backends.pytorch import loop

        net = torch.nn.Linear(3, 1, bias=False)
        torch.nn.init.constant_(net.weight, 0.5)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        settings = {"epochs": 1, "gradAccumulation": accumulate, **hp}
        return net, loop.Stepper(optimizer, list(net.parameters()), "cpu", settings, steps_per_epoch)

    def test_two_half_batches_make_the_update_one_whole_batch_does(self):
        import torch

        x = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [2.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
        y = torch.tensor([[1.0], [0.0], [2.0], [1.5]])

        whole, stepper = self._stepper(1, 1)
        stepper.step(torch.nn.functional.mse_loss(whole(x), y))

        halves, stepper = self._stepper(2, 2)
        for part in (slice(0, 2), slice(2, 4)):
            stepper.step(torch.nn.functional.mse_loss(halves(x[part]), y[part]))
        torch.testing.assert_close(halves.weight, whole.weight)

    def test_nothing_is_updated_until_enough_batches_have_been_added_up(self):
        import torch

        net, stepper = self._stepper(3, 3)
        before = net.weight.detach().clone()
        for _ in range(2):
            stepper.step(net(torch.ones(1, 3)).sum())
        assert torch.equal(net.weight, before)
        stepper.step(net(torch.ones(1, 3)).sum())
        assert not torch.equal(net.weight, before)

    def test_the_batches_left_at_the_end_of_an_epoch_are_not_thrown_away(self):
        import torch

        net, stepper = self._stepper(2, 3)
        before = net.weight.detach().clone()
        for _ in range(3):  # one update after two batches; the third waits
            stepper.step(net(torch.ones(1, 3)).sum())
        after_two = net.weight.detach().clone()
        stepper.flush()
        assert not torch.equal(after_two, before) and not torch.equal(net.weight, after_two)
        settled = net.weight.detach().clone()
        stepper.flush()  # nothing is left, so this changes nothing
        assert torch.equal(net.weight, settled)

    def test_the_learning_rate_curve_counts_updates_not_batches(self):
        import torch

        net, stepper = self._stepper(4, 8, lrSchedule="linear")
        rates = []
        for _ in range(8):
            stepper.step(net(torch.ones(1, 3)).sum())
            rates.append(stepper.optimizer.param_groups[0]["lr"])
        # Eight batches are two updates, so a linear curve is whole until the first update (the fourth batch) and has
        # reached 0 right after it. A curve that counted batches would still be above 0.04 there.
        assert rates[:3] == [0.1, 0.1, 0.1]
        assert rates[3] == pytest.approx(0.0)
        assert rates[-1] == pytest.approx(0.0)


def test_the_loop_restores_only_the_weights_it_was_told_to_keep():
    import torch

    from omnitrain.plugins.backends.pytorch import loop

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(2, 2)
            self.lora_a = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.lora_a(self.frozen(x)).unsqueeze(1)

    net = Toy()
    original = net.frozen.weight.detach().clone()
    net.frozen.weight.requires_grad_(False)
    x = torch.randn(8, 2)
    y = torch.zeros(8, 1, dtype=torch.long)
    loader = [(x, y)]
    run = types.SimpleNamespace(check_abort=lambda: None, heartbeat=lambda: None, report=lambda *a: None)
    hp = {"epochs": 3, "patience": 3, "optimizer": "AdamW", "learningRate": 0.05}
    loop.fit(run, net, loader, loader, hp=hp, device="cpu", beat=lambda: None, objective=loop.Tagging(),
             span="test", keep=lambda name: name.startswith("lora_"))  # fmt: skip
    assert torch.equal(net.frozen.weight, original)
