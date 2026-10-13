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

from omnitrain.backends.base import TrainContext
from omnitrain.backends.yolo import YoloBackend, data, runtime
from omnitrain.backends.yolo import train as train_mod
from omnitrain.backends.yolo.compile import YoloHyperparameters, compile_yolo_config
from omnitrain.backends.yolo.model import YoloLoadedModel
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor

pytestmark = pytest.mark.skipif(not os.environ.get("YOLO_SMOKE"), reason="set YOLO_SMOKE=1 to train for real")

CLASSES = ["cat", "dog", "bird"]
# Enough for a network trained from scratch on a handful of images to respond to its input at all (see below).
EPOCHS = 15


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# Each class is a different colour. Pure noise would not do: pooling averages noise away, so even a random network
# gives the same output for every noise image, and a "the two pipelines agree" check then passes vacuously.
BASE_COLOURS = {"cat": (200, 40, 40), "dog": (40, 200, 40), "bird": (40, 40, 200)}


def coloured_image(label, rng, width=64, height=64):
    base = np.array(BASE_COLOURS[label], dtype=np.float32)
    noise = rng.normal(0, 25, (height, width, 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def test_train_load_predict_evaluate_and_export_with_the_real_library(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 6), ("validation", 3), ("test", 3)):
        for label in CLASSES:
            for i in range(count):
                item_id = f"{split}-{label}-{i}"
                coloured_image(label, rng).save(source / f"{item_id}.png")
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
        YoloHyperparameters(epochs=EPOCHS, batch_size=4, image_size=32, pretrained=False, learning_rate=0.01),
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
    # Early stopping (patience 10) may end training before EPOCHS; whatever ran is reported once per epoch, in order.
    epochs_run = [e for e, s, _ in reports if s == "validation"]
    assert epochs_run == list(range(1, len(epochs_run) + 1)) and 1 <= len(epochs_run) <= EPOCHS
    assert [(e, s) for e, s, _ in reports] == [(e, s) for e in epochs_run for s in ("train", "validation")]
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
    pixels = (
        np.asarray(Image.open(image).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )
    logits = session.run(None, {session.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None]})[0].reshape(-1)
    assert not logits.sum() == pytest.approx(1.0)  # raw logits, not probabilities
    np.testing.assert_allclose(softmax(logits), probs, atol=1e-3)

    # -- the export manifest reads the same class order the model itself reports
    meta = loaded.meta
    manifest = YoloBackend.preprocessing_manifest("smoke", config, meta)
    assert manifest["outputs"][0]["classes"] == loaded.output.labels

    # -- the generated Python devkit client, driven only by model.onnx + preprocessing.json, must agree with
    # Ultralytics' own prediction on a NON-square image (where cropping and squashing give different pixels)
    import json
    import types

    import omnitrain.export

    client_path = Path(omnitrain.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"
    # Compiled from source, not imported: an import would leave a __pycache__ inside the templates directory.
    client_module = types.ModuleType("generated_client")
    client_module.__file__ = str(client_path)  # the template locates model.onnx next to itself
    exec(compile(client_path.read_text(encoding="utf-8"), str(client_path), "exec"), client_module.__dict__)  # noqa: S102

    # Red bands on the sides and blue in the middle: the centre crop sees almost only blue, the squashed whole
    # image sees about half red, so a model that tells the colours apart answers differently for the two.
    wide_pixels = np.zeros((60, 90, 3), dtype=np.uint8)
    wide_pixels[:, :] = BASE_COLOURS["cat"]
    wide_pixels[:, 22:68] = BASE_COLOURS["bird"]
    wide = tmp_path / "wide.png"
    Image.fromarray(wide_pixels).save(wide)
    in_process = loaded.predict(pd.DataFrame({"image_path": [str(wide)]}))["class_probabilities"].iloc[0]

    # Guard against a vacuous comparison: the model must respond to its input at all.
    reds = tmp_path / "reds.png"
    coloured_image("cat", rng).save(reds)
    blues = tmp_path / "blues.png"
    coloured_image("bird", rng).save(blues)
    red_p, blue_p = (
        np.array(loaded.predict(pd.DataFrame({"image_path": [str(p)]}))["class_probabilities"].iloc[0])
        for p in (reds, blues)
    )
    assert np.abs(red_p - blue_p).max() > 1e-3, f"model ignores its input: {red_p} vs {blue_p}"

    def worst_difference(manifest_for_client: dict) -> float:
        """Largest per-class gap between the generated client and Ultralytics on the wide image."""
        (tmp_path / "preprocessing.json").write_text(json.dumps(manifest_for_client))
        client = client_module.OmniTrainClient(model_path=path, preprocessing_path=tmp_path / "preprocessing.json")
        # Unrounded: the client's own rounding to 4 decimals would hide small differences.
        pixels = client._preprocess_image(wide)
        logits = client.session.run(None, {client.session.get_inputs()[0].name: pixels})[0].reshape(-1)
        return float(np.abs(softmax(logits) - np.array(in_process)).max())

    assert worst_difference(manifest) < 1e-5

    # Negative control: without the crop mode the client squashes, and must be measurably off. If this stopped
    # holding, the assertion above would no longer prove anything about the crop.
    squashing = json.loads(json.dumps(manifest))
    del squashing["inputs"][0]["imageResize"]
    assert worst_difference(squashing) > 1e-4

    loaded.close()
    model.close()


def test_detection_trains_serves_evaluates_and_exports_with_the_real_library(tmp_path, monkeypatch):
    """Quality is not the point (a few epochs on a toy set learn little): every stage must run for real and produce
    the right shapes, with class names, boxes and the ONNX output layout as the rest of the system expects them."""
    import json

    classes = ["red square", "green: square/odd"]  # real names with spaces and punctuation, as class names are
    colours = [(230, 40, 40), (40, 200, 60)]
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 24), ("validation", 8), ("test", 6)):
        for i in range(count):
            pixels = np.full((96, 128, 3), 30, dtype=np.uint8)
            boxes = []
            for _ in range(int(rng.integers(0, 3))):  # zero boxes is a background image
                c = int(rng.integers(0, 2))
                w, h = int(rng.integers(20, 40)), int(rng.integers(20, 40))
                x, y = int(rng.integers(0, 128 - w)), int(rng.integers(0, 96 - h))
                pixels[y : y + h, x : x + w] = colours[c]
                boxes.append({"label": classes[c], "x": x / 128, "y": y / 96, "w": w / 128, "h": h / 96})
            item_id = f"{split}-{i}"
            Image.fromarray(pixels).save(source / f"{item_id}.png")
            rows.append(
                {
                    "image_path": f"s3://theseus-datasets/pool/{item_id}.png",
                    "boxes": json.dumps(boxes),
                    "split": split,
                    "_theseus_item_id": item_id,
                }
            )
    # One image nobody labelled: it must be left out of training and evaluation, not treated as empty.
    Image.fromarray(np.full((96, 128, 3), 30, dtype=np.uint8)).save(source / "unlabelled.png")
    rows.append(
        {
            "image_path": "s3://theseus-datasets/pool/unlabelled.png",
            "boxes": None,
            "split": "train",
            "_theseus_item_id": "unlabelled",
        }
    )
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

    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="image_path", kind="storage_uri"),
            ColumnSpec(name="boxes", kind="boxes"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=classes,
    )
    config = compile_yolo_config(
        get_task_descriptor("object_detection"),
        snapshot,
        YoloHyperparameters(epochs=3, batch_size=8, image_size=128, pretrained=False),
    )
    reports = []
    ctx = TrainContext(
        run_id="smoke-detect",
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
    assert isinstance(model, YoloLoadedModel) and not (tmp_path / "work").exists()
    assert sorted(p.name for p in uploaded.iterdir()) == ["args.yaml", "best.pt", "names.json", "results.csv"]
    assert [(e, s) for e, s, _ in reports] == [(e, s) for e in (1, 2, 3) for s in ("train", "validation")]
    first_validation = reports[1][2]
    assert {"loss", "box_loss", "cls_loss", "dfl_loss", "precision", "recall", "mAP50", "mAP50-95"} <= set(
        first_validation
    )
    assert first_validation["loss"] == pytest.approx(
        first_validation["box_loss"] + first_validation["cls_loss"] + first_validation["dfl_loss"]
    )
    meta = json.loads((uploaded / "names.json").read_text())
    assert meta["task"] == "detect" and meta["names"] == classes  # the real names, in index order
    assert model.output.kind == "detections" and model.output.labels == classes

    # -- load back from what was uploaded, and predict
    loaded = YoloBackend.load(str(uploaded))
    probe = source / "test-0.png"
    frame = loaded.predict(pd.DataFrame({"image_path": [str(probe)]}), confidence=0.001)
    detections = frame["boxes_predictions"].iloc[0]
    assert len(detections) > 0  # at a threshold of 0.001 even an untrained detector proposes something
    for d in detections:
        assert d["label"] in classes and 0.0 <= d["confidence"] <= 1.0
        box = d["box"]
        assert (
            0 <= box["x"] <= 1
            and 0 <= box["y"] <= 1
            and box["x"] + box["w"] <= 1 + 1e-6
            and box["y"] + box["h"] <= 1 + 1e-6
        )
    assert [d["confidence"] for d in detections] == sorted((d["confidence"] for d in detections), reverse=True)
    assert list(frame["boxes_image_size"].iloc[0]) == [128, 96]
    out = loaded.to_output(frame)
    assert out["kind"] == "detections" and (out["imageWidth"], out["imageHeight"]) == (128, 96)

    # -- evaluate: scored by the real validator, on the labelled test images only
    result = YoloBackend.evaluate(loaded, df, "split", "_theseus_item_id")
    assert result is not None
    report = result.report
    assert report["kind"] == "detection" and report["split"] == "test" and report["imageCount"] == 6
    assert set(report["overall"]) == {"precision", "recall", "map50", "map5095"}
    assert set(report["perClass"]) == set(classes)
    assert report["instanceCount"] == sum(c["instances"] for c in report["perClass"].values())
    assert len(result.predictions) == 6 and "unlabelled" not in set(result.predictions["_theseus_item_id"])

    # -- convert: the raw candidate layout the manifest promises, with classes in the manifest in model order
    import onnxruntime as ort

    path = YoloBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    session = ort.InferenceSession(path)
    output = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 128, 128), dtype=np.float32)})[0]
    anchors = (128 // 8) ** 2 + (128 // 16) ** 2 + (128 // 32) ** 2  # the three detection strides
    assert output.shape == (1, 4 + len(classes), anchors)
    manifest = YoloBackend.preprocessing_manifest("smoke-detect", config, loaded.meta)
    assert manifest["outputs"][0]["classes"] == classes
    assert manifest["inputs"][0]["imageResize"]["mode"] == "letterbox"

    loaded.close()
    model.close()
