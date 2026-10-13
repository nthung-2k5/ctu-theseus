"""Trains a real (tiny, randomly initialised) video classifier on the CPU and runs it through the whole backend:
train, load, predict, evaluate, ONNX conversion. Opt-in, because it takes a while and imports torch for real:

    VIDEO_SMOKE=1 uv run python -m pytest tests/test_video_smoke.py

It needs no network: `pretrained=False` skips the Kinetics download and the clips are synthetic videos written to a
temp directory. S3 is replaced by the local file system.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnitrain.backends.base import TrainContext
from omnitrain.backends.video import VideoBackend, data, frames, runtime
from omnitrain.backends.video import train as train_mod
from omnitrain.backends.video.compile import VideoHyperparameters, compile_video_config
from omnitrain.backends.video.model import VideoLoadedModel
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

pytestmark = pytest.mark.skipif(not os.environ.get("VIDEO_SMOKE"), reason="set VIDEO_SMOKE=1 to train for real")

CLASSES = ["red", "green", "blue"]
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}  # RGB
EPOCHS = 6


def write_clip(path: Path, label: str, rng: np.random.Generator, frame_count: int = 12) -> None:
    """A clip of one flat colour per class with a little noise and a drifting brighter bar."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 32))
    r, g, b = COLOURS[label]
    for i in range(frame_count):
        frame = np.zeros((32, 48, 3), dtype=np.float32)
        frame[:, :] = (b, g, r)  # OpenCV writes BGR
        frame += rng.normal(0, 20, frame.shape)
        frame[:, (i * 3) % 40 : (i * 3) % 40 + 6] *= 0.6
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()


def test_train_load_predict_evaluate_and_export_with_the_real_library(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 6), ("validation", 3), ("test", 3)):
        for label in CLASSES:
            for i in range(count):
                item_id = f"{split}-{label}-{i}"
                write_clip(source / f"{item_id}.mp4", label, rng)
                rows.append(
                    {
                        "video_path": f"s3://theseus-datasets/pool/{item_id}.mp4",
                        "class": label,
                        "split": split,
                        "_theseus_item_id": item_id,
                    }
                )
    df = pd.DataFrame(rows)

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(source / os.path.basename(key), local_path)
        return local_path

    uploaded = tmp_path / "uploaded"

    def fake_upload_dir(bucket, prefix, local_dir, **kw):
        shutil.copytree(local_dir, uploaded)

    monkeypatch.setattr(data.storage, "download_file", fake_download)
    monkeypatch.setattr(train_mod.storage, "upload_dir", fake_upload_dir)
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="video_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=CLASSES,
    )
    config = compile_video_config(
        get_task_descriptor("video_classification"),
        snapshot,
        VideoHyperparameters(
            epochs=EPOCHS, batch_size=4, num_frames=8, frame_stride=1, frame_size=32, pretrained=False,
            learning_rate=0.003, clips_per_video=2, seed=1,
        ),
    )  # fmt: skip
    reports = []
    ctx = TrainContext(
        run_id="smoke",
        config=config,
        dataset_uri="unused",
        output_uri="unused",
        workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None,
        _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )

    # -- train
    model = VideoBackend.train(ctx)
    assert isinstance(model, VideoLoadedModel)
    assert not (tmp_path / "work").exists()
    assert sorted(p.name for p in uploaded.iterdir()) == ["model.pt", "names.json"]
    epochs_run = [e for e, s, _ in reports if s == "validation"]
    assert epochs_run == list(range(1, len(epochs_run) + 1)) and 1 <= len(epochs_run) <= EPOCHS
    assert [(e, s) for e, s, _ in reports] == [(e, s) for e in epochs_run for s in ("train", "validation")]
    assert all("loss" in m and "accuracy" in m for _, _, m in reports)
    # The class order is the one the model was trained with, written into the artifact itself.
    assert model.output.labels == CLASSES

    # -- load back from what was uploaded
    loaded = VideoBackend.load(str(uploaded))
    assert loaded.output.labels == CLASSES

    # -- predict
    clip = source / "test-green-0.mp4"
    frame = loaded.predict(pd.DataFrame({"video_path": [str(clip)]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    out = loaded.to_output(frame)
    assert out["kind"] == "classification" and len(out["classes"]) == 3

    # -- evaluate
    result = VideoBackend.evaluate(loaded, df, "split", "_theseus_item_id")
    assert result is not None
    assert result.report["split"] == "test" and result.report["rowCount"] == 9
    assert np.array(result.report["confusionMatrix"]).sum() == 9
    assert result.report["idx2str"] == CLASSES

    # -- convert: the ONNX graph must reproduce the in-process logits for the same clip tensor
    import onnxruntime as ort
    import torch

    path = VideoBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    assert Path(path).name == "model.onnx"
    session = ort.InferenceSession(path)
    assert session.get_inputs()[0].name == "clips" and session.get_inputs()[0].shape[1:] == [3, 8, 32, 32]
    [array] = frames.load_clips(str(clip), loaded.spec, clips=1)
    tensor = frames.to_tensor(array, loaded.spec)[None]
    with torch.no_grad():
        expected = loaded.net(tensor.to(runtime.device())).cpu().numpy()
    (logits,) = session.run(["logits"], {"clips": tensor.numpy()})
    np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-3)
