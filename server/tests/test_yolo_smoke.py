"""Trains a real (tiny, randomly initialised) YOLO classifier on the CPU and runs it through the whole backend:
train, load, predict, evaluate, ONNX conversion. Opt-in, because it takes about half a minute and imports
torch and ultralytics for real:

    YOLO_SMOKE=1 uv run python -m pytest tests/test_yolo_smoke.py

It needs no network: the architecture comes from a `.yaml` bundled with Ultralytics (`pretrained=False`) and
the images are random noise written to a temp directory. S3 is replaced by the local file system.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from theseus.backends.base import TrainContext
from theseus.backends.yolo import YoloBackend, data, runtime
from theseus.backends.yolo import train as train_mod
from theseus.backends.yolo.compile import YoloHyperparameters, compile_yolo_config
from theseus.backends.yolo.model import YoloLoadedModel
from theseus.services.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

pytestmark = pytest.mark.skipif(not os.environ.get("YOLO_SMOKE"), reason="set YOLO_SMOKE=1 to train for real")

CLASSES = ["cat", "dog", "bird"]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def test_train_load_predict_evaluate_and_export_with_the_real_library(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 6), ("validation", 3), ("test", 3)):
        for label in CLASSES:
            for i in range(count):
                item_id = f"{split}-{label}-{i}"
                Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(source / f"{item_id}.png")
                rows.append(
                    {
                        "image_path": f"s3://theseus-datasets/pool/{item_id}.png",
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

    ctx_snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="image_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=CLASSES,
    )
    config = compile_yolo_config(
        get_task_descriptor("image_classification"),
        ctx_snapshot,
        YoloHyperparameters(epochs=2, batch_size=4, image_size=32, pretrained=False),
    )
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
    model = YoloBackend.train(ctx)
    assert isinstance(model, YoloLoadedModel)
    assert not (tmp_path / "work").exists()
    assert sorted(p.name for p in uploaded.iterdir()) == ["args.yaml", "best.pt", "names.json", "results.csv"]
    assert [(e, s) for e, s, _ in reports] == [(1, "train"), (1, "validation"), (2, "train"), (2, "validation")]
    assert all("loss" in m for _, _, m in reports) and "accuracy" in reports[1][2]
    # Whatever order the model learned, labels are real class names and a permutation of the snapshot's.
    assert sorted(model.output.labels or []) == sorted(CLASSES)

    # -- load back from what was uploaded
    loaded = YoloBackend.load(str(uploaded))
    assert loaded.output.labels == model.output.labels

    # -- predict
    image = source / "test-cat-0.png"
    frame = loaded.predict(pd.DataFrame({"image_path": [str(image)]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    out = loaded.to_output(frame)
    assert out["kind"] == "classification" and len(out["classes"]) == 3

    # -- evaluate
    result = YoloBackend.evaluate(loaded, df, "split", "_theseus_item_id")
    assert result is not None
    assert result.report["split"] == "test" and result.report["rowCount"] == 9
    assert np.array(result.report["confusionMatrix"]).sum() == 9

    # -- convert: the ONNX graph must reproduce the in-process probabilities once a client softmaxes its logits
    import onnxruntime as ort

    path = YoloBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    assert Path(path).name == "model.onnx"
    session = ort.InferenceSession(path)
    pixels = np.asarray(Image.open(image).convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
    logits = session.run(None, {session.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None]})[0].reshape(-1)
    assert not logits.sum() == pytest.approx(1.0)  # raw logits, not probabilities
    np.testing.assert_allclose(softmax(logits), probs, atol=1e-3)

    # -- the export manifest reads the same class order the model itself reports
    meta = loaded.meta
    manifest = YoloBackend.preprocessing_manifest("smoke", config, meta)
    assert manifest["outputs"][0]["classes"] == loaded.output.labels

    loaded.close()
    model.close()
