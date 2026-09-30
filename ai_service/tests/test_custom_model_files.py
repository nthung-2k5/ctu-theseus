"""Bring-your-own models without a database: safe unpacking, the generic file checks, the Ludwig-specific
validation, and how each custom kind compiles into a Ludwig config."""

import json
import stat
import zipfile
from pathlib import Path

import pytest

from theseus.backends.base import ConfigError, CustomModelRef
from theseus.backends.ludwig.compile import LudwigHyperparameters, compile_ludwig_config
from theseus.backends.ludwig.custom import validate_custom_model
from theseus.services import custom_models as cm
from theseus.services.task_registry import SnapshotContext, get_task_descriptor

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


# -- The Ludwig backend's own validation -------------------------------------------------------


class TestValidateCustomModel:
    text = get_task_descriptor("text_classification")
    llm = get_task_descriptor("text_generation")

    def test_a_good_encoder_passes(self, tmp_path):
        validate_custom_model(self.text, ref("hf_transformer", str(write_hf_model(tmp_path))))

    def test_a_kind_the_task_does_not_offer_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot be used"):
            validate_custom_model(self.llm, ref("hf_transformer", str(write_hf_model(tmp_path))))

    def test_missing_or_broken_config_json(self, tmp_path):
        with pytest.raises(ConfigError, match="no config.json"):
            validate_custom_model(self.text, ref("hf_transformer", str(tmp_path)))
        (tmp_path / "config.json").write_text("{not json")
        with pytest.raises(ConfigError, match="could not be read"):
            validate_custom_model(self.text, ref("hf_transformer", str(tmp_path)))

    def test_config_without_a_model_type_is_not_a_transformers_model(self, tmp_path):
        write_hf_model(tmp_path, model_type=None)
        with pytest.raises(ConfigError, match="model_type"):
            validate_custom_model(self.text, ref("hf_transformer", str(tmp_path)))

    def test_remote_code_is_refused(self, tmp_path):
        write_hf_model(tmp_path, extra_config={"auto_map": {"AutoModel": "modeling.Model"}})
        with pytest.raises(ConfigError, match="custom code"):
            validate_custom_model(self.text, ref("hf_transformer", str(tmp_path)))

    def test_weights_must_be_safetensors(self, tmp_path):
        write_hf_model(tmp_path, weights=False)
        with pytest.raises(ConfigError, match="safetensors"):
            validate_custom_model(self.text, ref("hf_transformer", str(tmp_path)))

    def test_a_causal_lm_must_actually_be_one(self, tmp_path):
        write_hf_model(tmp_path, architectures=["LlamaForCausalLM"], model_type="llama")
        validate_custom_model(self.llm, ref("hf_causal_lm", str(tmp_path)))
        write_hf_model(tmp_path / "enc", architectures=["BertModel"])
        with pytest.raises(ConfigError, match="Not a causal language model"):
            validate_custom_model(self.llm, ref("hf_causal_lm", str(tmp_path / "enc")))

    def test_a_gpt2_style_name_counts_as_causal(self, tmp_path):
        write_hf_model(tmp_path, architectures=["GPT2LMHeadModel"], model_type="gpt2")
        validate_custom_model(self.llm, ref("hf_causal_lm", str(tmp_path)))

    def test_timm_needs_a_name_and_the_package(self):
        image = get_task_descriptor("image_classification")
        with pytest.raises(ConfigError, match="model name"):
            validate_custom_model(image, ref("timm_image", source_ref=None))
        try:
            import timm  # noqa: F401
        except ImportError:
            with pytest.raises(ConfigError, match="not installed"):
                validate_custom_model(image, ref("timm_image", source_ref="resnet50.a1_in1k"))
        else:
            with pytest.raises(ConfigError, match="not a timm architecture"):
                validate_custom_model(image, ref("timm_image", source_ref="definitely_not_a_model"))


# -- Compiling ---------------------------------------------------------------------------------


def compile_custom(task_id: str, custom: CustomModelRef | None, ctx: SnapshotContext | None = None, **sel):
    hp = LudwigHyperparameters(encoderId="custom:1", **sel)
    return compile_ludwig_config(get_task_descriptor(task_id), ctx or SnapshotContext(), hp, custom)


class TestCompileCustom:
    def test_a_text_encoder_becomes_auto_transformer_at_its_local_path(self):
        config = compile_custom("text_classification", ref("hf_transformer", "/m/bert"))
        assert config["input_features"][0]["encoder"] == {
            "type": "auto_transformer",
            "pretrained_model_name_or_path": "/m/bert",
        }
        assert config["model_type"] == "ecd" and "base_model" not in config

    def test_freezing_a_custom_encoder(self):
        config = compile_custom("text_classification", ref("hf_transformer", "/m/bert"), freezeBackbone=True)
        assert config["input_features"][0]["encoder"]["trainable"] is False

    def test_a_causal_lm_becomes_the_base_model_and_the_key_survives_validation(self):
        config = compile_custom("text_generation", ref("hf_causal_lm", "/m/llama"))
        assert config["model_type"] == "llm" and config["base_model"] == "/m/llama"
        assert all("encoder" not in f for f in config["input_features"])

    def test_a_builtin_llm_task_still_compiles_without_a_base_model(self):
        # The pre-existing behaviour: nothing forces a base model, so this does not become an error here.
        config = compile_ludwig_config(
            get_task_descriptor("text_generation"), SnapshotContext(), LudwigHyperparameters()
        )
        assert "base_model" not in config

    def test_freezing_a_language_model_is_refused(self):
        with pytest.raises(ConfigError, match="language model"):
            compile_custom("text_generation", ref("hf_causal_lm", "/m/llama"), freezeBackbone=True)

    def test_a_timm_model_is_selected_by_name(self):
        config = compile_custom("image_classification", ref("timm_image", "/unused", source_ref="resnet50.a1_in1k"))
        assert config["input_features"][0]["encoder"] == {
            "type": "timm",
            "model_name": "resnet50.a1_in1k",
            "use_pretrained": True,
        }

    def test_image_options_still_apply_to_a_custom_encoder(self):
        config = compile_custom(
            "image_classification", ref("timm_image", "/unused", source_ref="resnet50"), imageSize=224
        )
        assert config["input_features"][0]["preprocessing"] == {"height": 224, "width": 224}

    def test_a_kind_the_task_does_not_offer_is_refused(self):
        with pytest.raises(ConfigError, match="cannot be used for task"):
            compile_custom("image_classification", ref("hf_transformer", "/m/bert"))

    def test_a_custom_id_that_was_never_resolved_is_refused_not_looked_up_as_a_builtin(self):
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_custom("text_classification", None)

    def test_the_head_options_still_apply(self):
        config = compile_custom("text_classification", ref("hf_transformer", "/m/bert"), headLayers=2)
        assert config["combiner"]["num_fc_layers"] == 2


# -- Hugging Face vision models (hf_vision) ----------------------------------------------------


def write_tiny_vit(root: Path) -> Path:
    from transformers import ViTConfig, ViTModel

    ViTModel(
        ViTConfig(
            image_size=32,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_channels=3,
        )  # fmt: skip
    ).save_pretrained(root)
    return root


class TestHfVision:
    image = get_task_descriptor("image_classification")

    def test_it_is_offered_for_image_classification_only(self):
        from theseus.backends.ludwig.tasks import custom_kinds_for

        assert "hf_vision" in {k.id for k in custom_kinds_for("image_classification")}
        for task in ("image_captioning", "text_classification", "text_generation", "tabular_classification"):
            assert "hf_vision" not in {k.id for k in custom_kinds_for(task)}, task

    def test_it_compiles_to_the_backends_own_encoder(self):
        config = compile_custom("image_classification", ref("hf_vision", "/m/vit"), freezeBackbone=True, imageSize=224)
        feature = config["input_features"][0]
        assert feature["encoder"] == {
            "type": "hf_vision",
            "pretrained_model_name_or_path": "/m/vit",
            "trainable": False,
        }
        assert feature["preprocessing"] == {"height": 224, "width": 224}

    def test_it_is_refused_for_a_task_that_does_not_offer_it(self):
        with pytest.raises(ConfigError, match="cannot be used for task"):
            compile_custom("text_classification", ref("hf_vision", "/m/vit"))

    def test_a_real_vision_backbone_passes(self, tmp_path):
        validate_custom_model(self.image, ref("hf_vision", str(write_tiny_vit(tmp_path))))

    def test_a_text_model_is_not_an_image_backbone(self, tmp_path):
        from transformers import BertConfig, BertModel

        BertModel(
            BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        ).save_pretrained(tmp_path)
        with pytest.raises(ConfigError, match="cannot be loaded as an image backbone"):
            validate_custom_model(self.image, ref("hf_vision", str(tmp_path)))

    def test_a_checkpoint_with_a_text_tower_is_refused_with_advice(self, tmp_path):
        write_hf_model(tmp_path, model_type="clip", extra_config={"vision_config": {}, "text_config": {}})
        with pytest.raises(ConfigError, match="vision-only checkpoint"):
            validate_custom_model(self.image, ref("hf_vision", str(tmp_path)))

    def test_the_generic_folder_rules_still_apply(self, tmp_path):
        vit = write_tiny_vit(tmp_path / "vit")
        (vit / "model.safetensors").unlink()
        with pytest.raises(ConfigError, match="safetensors"):
            validate_custom_model(self.image, ref("hf_vision", str(vit)))
        with pytest.raises(ConfigError, match="no config.json"):
            validate_custom_model(self.image, ref("hf_vision", str(tmp_path / "empty")))
        remote = write_tiny_vit(tmp_path / "remote")
        config = json.loads((remote / "config.json").read_text())
        (remote / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "x.Y"}}))
        with pytest.raises(ConfigError, match="custom code"):
            validate_custom_model(self.image, ref("hf_vision", str(remote)))
