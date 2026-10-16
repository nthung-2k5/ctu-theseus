"""Bring-your-own Hugging Face video models (VideoMAE and friends): kinds, file checks, compile, and a real tiny
train / load / predict / evaluate / ONNX round trip. The model is a randomly initialised 1-layer VideoMAE written to a
temp folder, so nothing is downloaded."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")

from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor  # noqa: E402
from omnitrain.plugins.backends.base import ConfigError, CustomModelRef, TrainContext  # noqa: E402
from omnitrain.plugins.backends.registry import get_backend  # noqa: E402
from omnitrain.plugins.backends.video import VideoBackend, custom, data, frames, runtime  # noqa: E402
from omnitrain.plugins.backends.video import train as train_mod  # noqa: E402
from omnitrain.plugins.backends.video.compile import VideoHyperparameters, compile_video_config  # noqa: E402
from omnitrain.plugins.backends.video.model import VideoLoadedModel  # noqa: E402

TASK = get_task_descriptor("video_classification")
CLASSES = ["red", "green", "blue"]
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
SNAPSHOT = SnapshotContext(
    columns=[
        ColumnSpec(name="video_path", kind="storage_uri"),
        ColumnSpec(name="class", kind="label"),
        ColumnSpec(name="split", kind="split"),
    ],
    label_class_names=CLASSES,
)


def make_tiny_videomae(path: Path, *, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)) -> Path:
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    config = VideoMAEConfig(
        image_size=32, patch_size=16, num_frames=4, tubelet_size=2, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64, num_labels=5,
    )  # fmt: skip
    path.mkdir(parents=True, exist_ok=True)
    VideoMAEForVideoClassification(config).save_pretrained(path, safe_serialization=True)
    (path / "preprocessor_config.json").write_text(json.dumps({"image_mean": list(mean), "image_std": list(std)}))
    return path


def ref_for(path: Path, kind: str = custom.HF_VIDEO) -> CustomModelRef:
    return CustomModelRef(id="u1", kind=kind, source_kind="upload", local_path=str(path))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return make_tiny_videomae(tmp_path_factory.mktemp("videomae"))


class TestKindAndValidation:
    def test_the_kind_is_offered_for_video_classification_only(self):
        [kind] = VideoBackend.custom_model_kinds(TASK)
        assert (kind.id, kind.modality, kind.source_kinds, kind.status) == (
            "hf_video",
            "video",
            ["hub", "upload"],
            "experimental",
        )
        assert kind.unavailable_reason is None
        assert VideoBackend.custom_model_kinds(get_task_descriptor("image_classification")) == []

    @pytest.mark.parametrize(
        "config, expected",
        [
            ({"model_type": "videomae", "architectures": ["VideoMAEForPreTraining"]}, "hf_video"),
            ({"model_type": "videomae", "architectures": ["VideoMAEForVideoClassification"]}, "hf_video"),
            ({"model_type": "timesformer", "architectures": ["TimesformerModel"]}, "hf_video"),
            ({"model_type": "vivit", "architectures": ["VivitForVideoClassification"]}, "hf_video"),
            ({"model_type": "vit", "architectures": ["ViTForImageClassification"]}, None),
            ({"model_type": "bert", "architectures": ["BertForSequenceClassification"]}, None),
            ({"model_type": "videomae"}, None),
        ],
    )
    def test_the_architecture_says_which_checkpoints_are_video_models(self, config, expected):
        assert VideoBackend.suggest_custom_kind(config) == expected

    def test_a_good_model_validates(self, tiny):
        VideoBackend.validate_custom_model(TASK, ref_for(tiny))

    def test_a_kind_that_is_not_ours_is_refused(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            VideoBackend.validate_custom_model(TASK, ref_for(tiny, kind="hf_vision"))

    def test_code_that_runs_from_the_repo_is_refused(self, tiny, tmp_path):
        folder = tmp_path / "m"
        shutil.copytree(tiny, folder)
        config = json.loads((folder / "config.json").read_text())
        (folder / "config.json").write_text(json.dumps({**config, "auto_map": {"AutoModel": "x.Y"}}))
        with pytest.raises(ConfigError, match="custom code"):
            VideoBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_without_safetensors_is_refused(self, tiny, tmp_path):
        folder = tmp_path / "m"
        shutil.copytree(tiny, folder)
        (folder / "model.safetensors").unlink()
        (folder / "pytorch_model.bin").write_bytes(b"pickle")
        with pytest.raises(ConfigError, match="safetensors"):
            VideoBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_model_of_another_family_is_refused(self, tiny, tmp_path):
        folder = tmp_path / "m"
        shutil.copytree(tiny, folder)
        config = json.loads((folder / "config.json").read_text())
        (folder / "config.json").write_text(json.dumps({**config, "model_type": "bert"}))
        with pytest.raises(ConfigError, match="supported video model"):
            VideoBackend.validate_custom_model(TASK, ref_for(folder))

    def test_weights_that_do_not_match_the_config_are_refused(self, tiny, tmp_path):
        folder = tmp_path / "m"
        shutil.copytree(tiny, folder)
        config = json.loads((folder / "config.json").read_text())
        (folder / "config.json").write_text(json.dumps({**config, "hidden_size": 64, "intermediate_size": 128}))
        with pytest.raises(ConfigError, match="cannot be loaded as a video classifier"):
            VideoBackend.validate_custom_model(TASK, ref_for(folder))

    def test_a_folder_that_is_not_a_model_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="config.json"):
            VideoBackend.validate_custom_model(TASK, ref_for(tmp_path))


class TestClipSettings:
    def test_frames_and_size_come_from_the_config_and_normalisation_from_the_preprocessor(self, tiny):
        assert custom.clip_settings(tiny) == {
            "numFrames": 4, "frameSize": 32, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]
        }  # fmt: skip

    def test_without_a_preprocessor_file_it_is_imagenet(self, tiny, tmp_path):
        folder = tmp_path / "m"
        shutil.copytree(tiny, folder)
        (folder / "preprocessor_config.json").unlink()
        assert custom.clip_settings(folder)["mean"] == [0.485, 0.456, 0.406]


class TestCompile:
    def test_a_custom_model_is_recorded_for_training_and_ignores_the_clip_knobs(self, tiny):
        config = compile_video_config(
            TASK,
            SNAPSHOT,
            VideoHyperparameters(model_id="custom:u1", num_frames=8, frame_size=64),
            custom=ref_for(tiny),
        )
        assert config["model"] == "custom:u1" and config["pretrained"] is True
        assert config["custom"] == {"kind": "hf_video", "path": str(tiny)}
        assert config["classNames"] == CLASSES

    def test_an_unresolved_custom_id_is_an_error(self):
        with pytest.raises(ConfigError, match="was not resolved"):
            compile_video_config(TASK, SNAPSHOT, VideoHyperparameters(model_id="custom:gone"))

    def test_a_custom_model_of_the_wrong_kind_is_an_error(self, tiny):
        with pytest.raises(ConfigError, match="cannot be used"):
            compile_video_config(
                TASK, SNAPSHOT, VideoHyperparameters(model_id="custom:u1"), custom=ref_for(tiny, kind="timm_image")
            )

    def test_a_built_in_model_has_no_custom_block(self):
        assert "custom" not in compile_video_config(TASK, SNAPSHOT, VideoHyperparameters())


def write_clip(path: Path, label: str, rng: np.random.Generator) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 32))
    r, g, b = COLOURS[label]
    for _ in range(10):
        frame = np.zeros((32, 48, 3), dtype=np.float32)
        frame[:, :] = (b, g, r)
        writer.write(np.clip(frame + rng.normal(0, 20, frame.shape), 0, 255).astype(np.uint8))
    writer.release()


def test_fine_tune_load_predict_evaluate_and_export_a_custom_model(tiny, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 6), ("validation", 3), ("test", 3)):
        for label in CLASSES:
            for i in range(count):
                item = f"{split}-{label}-{i}"
                write_clip(source / f"{item}.mp4", label, rng)
                rows.append(
                    {"video_path": f"s3://theseus-datasets/pool/{item}.mp4", "class": label, "split": split,
                     "_theseus_item_id": item}
                )  # fmt: skip
    df = pd.DataFrame(rows)

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(source / os.path.basename(key), local_path)
        return local_path

    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(data.storage, "download_file", fake_download)
    monkeypatch.setattr(train_mod.storage, "upload_dir", lambda b, p, d, **kw: shutil.copytree(d, uploaded))
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    # A private copy of the model's files, deleted afterwards to prove the trained model does not need them.
    original = tmp_path / "original"
    shutil.copytree(tiny, original)
    config = compile_video_config(
        TASK, SNAPSHOT,
        VideoHyperparameters(model_id="custom:u1", epochs=3, batch_size=4, frame_stride=1, clips_per_video=2, seed=1),
        custom=ref_for(original),
    )  # fmt: skip
    reports = []
    ctx = TrainContext(
        run_id="hf", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )  # fmt: skip

    # -- train
    model = VideoBackend.train(ctx)
    assert isinstance(model, VideoLoadedModel)
    assert {"config.json", "model.safetensors", "names.json"} <= {p.name for p in uploaded.iterdir()}
    assert not (uploaded / "model.pt").exists()
    meta = json.loads((uploaded / "names.json").read_text())
    # The model dictated the clip, whatever the config asked for.
    assert (meta["hf"], meta["numFrames"], meta["frameSize"], meta["mean"], meta["names"]) == (
        True, 4, 32, [0.5, 0.5, 0.5], CLASSES
    )  # fmt: skip
    assert model.output.labels == CLASSES  # the new 3-class head, not the 5-class one the checkpoint came with
    assert all("loss" in m and "accuracy" in m for _, _, m in reports) and {s for _, s, _ in reports} == {
        "train",
        "validation",
    }

    # -- load back from what was uploaded, with the original files gone
    shutil.rmtree(original)
    loaded = VideoBackend.load(str(uploaded))
    assert loaded.output.labels == CLASSES and loaded.spec.num_frames == 4 and loaded.spec.size == 32

    # -- predict and evaluate
    clip = source / "test-green-0.mp4"
    frame = loaded.predict(pd.DataFrame({"video_path": [str(clip)]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    result = VideoBackend.evaluate(loaded, df, "split", "_theseus_item_id")
    assert result is not None and result.report["rowCount"] == 9 and result.report["idx2str"] == CLASSES

    # -- convert: the ONNX graph takes the same NCTHW clip tensor and reproduces the in-process logits
    import onnxruntime as ort
    import torch

    path = get_backend("video").convert(loaded, "onnx", str(tmp_path / "export"))
    session = ort.InferenceSession(path)
    assert session.get_inputs()[0].name == "clips" and session.get_inputs()[0].shape[1:] == [3, 4, 32, 32]
    [array] = frames.load_clips(str(clip), loaded.spec, clips=1)
    tensor = frames.to_tensor(array, loaded.spec)[None]
    with torch.no_grad():
        expected = loaded.net(tensor.to(runtime.device())).cpu().numpy()
    (logits,) = session.run(["logits"], {"clips": tensor.numpy()})
    np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-3)
