"""Trains a real (tiny, randomly initialised) image classifier on the CPU and runs it through the whole backend:
train, load, predict, evaluate, ONNX conversion, and the generated Python client. Opt-in, because it imports torch for
real and exports ONNX graphs:

    PYTORCH_SMOKE=1 uv run python -m pytest tests/test_pytorch_smoke.py

It needs no network: `pretrained=False` skips the ImageNet download and the images are synthetic files written to a
temp directory. S3 is replaced by the local file system.
"""

import json
import os
import shutil
import types
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import omnitrain.plugins.export
from omnitrain.domain.task_registry import ColumnSpec, SnapshotContext, get_task_descriptor
from omnitrain.plugins.backends.base import TrainContext
from omnitrain.plugins.backends.pytorch import PytorchBackend, data, runtime, tasks
from omnitrain.plugins.backends.pytorch import train as train_mod
from omnitrain.plugins.backends.pytorch.compile import PytorchHyperparameters, compile_pytorch_config
from omnitrain.plugins.backends.pytorch.manifest import build_preprocessing_manifest
from omnitrain.plugins.backends.pytorch.model import PytorchLoadedModel, wrap

pytestmark = pytest.mark.skipif(not os.environ.get("PYTORCH_SMOKE"), reason="set PYTORCH_SMOKE=1 to train for real")

CLASSES = ["red", "green", "blue"]
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
EPOCHS = 6
CLIENT_PATH = Path(omnitrain.plugins.export.__file__).parent / "templates" / "python" / "omnitrain_client.py"


def write_image(path: Path, label: str, rng: np.random.Generator) -> None:
    """A flat colour per class with noise, deliberately not square so the shorter-edge crop is exercised."""
    pixels = np.zeros((48, 72, 3), dtype=np.float32)
    pixels[:, :] = COLOURS[label]
    pixels += rng.normal(0, 25, pixels.shape)
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pytorch")
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 8), ("validation", 4), ("test", 4)):
        for label in CLASSES:
            for i in range(count):
                item_id = f"{split}-{label}-{i}"
                write_image(source / f"{item_id}.png", label, rng)
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

    mp = pytest.MonkeyPatch()
    mp.setattr(data.storage, "download_file", fake_download)
    mp.setattr(train_mod.storage, "upload_dir", fake_upload_dir)
    mp.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    mp.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")

    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="image_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=CLASSES,
    )
    config = compile_pytorch_config(
        get_task_descriptor("image_classification"),
        snapshot,
        PytorchHyperparameters(
            epochs=EPOCHS, batch_size=8, image_size=64, pretrained=False, learning_rate=0.003, seed=1,
            lr_schedule="cosine", warmup_epochs=1, grad_clip_norm=1.0,
        ),
    )  # fmt: skip
    reports: list = []
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
    model = PytorchBackend.train(ctx)
    yield types.SimpleNamespace(
        model=model, reports=reports, uploaded=uploaded, source=source, df=df, tmp_path=tmp_path, config=config
    )
    mp.undo()


def test_training_reports_each_epoch_and_uploads_the_artifacts(trained):
    assert isinstance(trained.model, PytorchLoadedModel)
    assert not (trained.tmp_path / "work").exists()
    assert sorted(p.name for p in trained.uploaded.iterdir()) == ["model.pt", "names.json"]
    epochs_run = [e for e, s, _ in trained.reports if s == "validation"]
    assert epochs_run == list(range(1, len(epochs_run) + 1)) and 1 <= len(epochs_run) <= EPOCHS
    assert [(e, s) for e, s, _ in trained.reports] == [(e, s) for e in epochs_run for s in ("train", "validation")]
    assert all("loss" in m and "accuracy" in m for _, _, m in trained.reports)
    # It learned something on three trivially separable colours.
    assert max(m["accuracy"] for _, s, m in trained.reports if s == "validation") > 0.6


def test_the_class_order_is_written_into_the_artifact(trained):
    meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
    assert meta["names"] == CLASSES and meta["imageSize"] == 64 and meta["imageColumn"] == "image_path"
    assert trained.model.output.labels == CLASSES
    assert PytorchBackend.load(str(trained.uploaded)).output.labels == CLASSES


def test_predict_and_evaluate_from_the_uploaded_files(trained):
    loaded = PytorchBackend.load(str(trained.uploaded))
    frame = loaded.predict(pd.DataFrame({"image_path": [str(trained.source / "test-green-0.png")]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    out = loaded.to_output(frame)
    assert out["kind"] == "classification" and len(out["classes"]) == 3

    result = PytorchBackend.evaluate(loaded, trained.df, "split", "_theseus_item_id")
    assert result is not None
    assert result.report["split"] == "test" and result.report["rowCount"] == 12
    assert np.array(result.report["confusionMatrix"]).sum() == 12
    assert result.report["idx2str"] == CLASSES


def test_the_onnx_export_matches_the_module_and_a_generated_client(trained, tmp_path):
    """The whole point of one `ImageSpec`: an image run through the shipped client (its own PIL preprocessing, the
    ONNX file, its own softmax) gives the platform's own probabilities."""
    import onnxruntime as ort
    import torch

    loaded = PytorchBackend.load(str(trained.uploaded))
    path = PytorchBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    assert Path(path).name == "model.onnx"
    # A bundle carries the model as one file, so the weights must be inside it and not in a sidecar.
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    session = ort.InferenceSession(path)
    assert session.get_inputs()[0].name == "images" and session.get_inputs()[0].shape[1:] == [3, 64, 64]

    # Straight logits parity, for a batch of two (the batch axis is dynamic).
    batch = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = loaded.net(batch.to(runtime.device())).cpu().numpy()
    (logits,) = session.run(["logits"], {"images": batch.numpy()})
    np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-3)

    # Through the generated client.
    module = types.ModuleType("omnitrain_client_smoke")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copyfile(path, bundle / "model.onnx")
    meta = json.loads((trained.uploaded / "names.json").read_text(encoding="utf-8"))
    (bundle / "preprocessing.json").write_text(json.dumps(build_preprocessing_manifest("smoke", {}, meta)))
    client = module.OmniTrainClient(bundle / "model.onnx", bundle / "preprocessing.json")
    image = trained.source / "test-blue-1.png"
    from_client = client.predict(str(image), top_k=3)
    frame = loaded.predict(pd.DataFrame({"image_path": [str(image)]}))
    from_platform = loaded.golden_prediction(frame)
    assert from_client.keys() == from_platform.keys()
    for label, p in from_client.items():
        assert p == pytest.approx(from_platform[label], abs=2e-3)


@pytest.mark.parametrize("model_id", [m.id for m in tasks.IMAGE_MODELS])
def test_every_catalog_model_builds_trains_a_step_and_exports(model_id, tmp_path):
    """Each architecture in the catalog (randomly initialised, no download) takes the new head, runs forward and
    backward, and exports to an ONNX graph that agrees with it."""
    import onnxruntime as ort
    import torch

    from omnitrain.plugins.backends.pytorch import network

    model = tasks.MODELS_BY_ID[model_id]
    size = model.size if model.fixed_size else 64
    net = network.build_model(model_id, 3, pretrained=False)
    net.train()
    loss = torch.nn.functional.cross_entropy(net(torch.randn(2, 3, size, size)), torch.tensor([0, 2]))
    loss.backward()
    assert loss.item() == loss.item()  # not NaN

    meta = {
        "model": model_id, "imageSize": size, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "imageColumn": "image_path", "outputName": "class", "names": CLASSES, "task": "classify", "inputKind": "image",
    }  # fmt: skip
    loaded = wrap(net, meta)
    path = PytorchBackend.convert(loaded, "onnx", str(tmp_path))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    batch = torch.randn(3, 3, size, size)
    with torch.no_grad():
        expected = loaded.net(batch.to(runtime.device())).cpu().numpy()
    (logits,) = ort.InferenceSession(path).run(["logits"], {"images": batch.numpy()})
    np.testing.assert_allclose(logits, expected, rtol=1e-2, atol=1e-2)


def test_freezing_the_backbone_trains_only_the_head():
    from omnitrain.plugins.backends.pytorch import network

    net = network.build_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


# -- tables --------------------------------------------------------------------------------------------------------


def make_table(rng: np.random.Generator, kind: str) -> pd.DataFrame:
    """Two numeric columns. A classifier's class depends on them (separable); a regressor's target is linear in them.
    Validation and test hold values far outside the train range, and a few cells are missing."""
    rows = []
    for split, count in (("train", 120), ("validation", 40), ("test", 40)):
        for i in range(count):
            a = float(rng.normal(0, 1)) + (50.0 if split == "test" and i % 5 == 0 else 0.0)
            b = float(rng.normal(100, 20))
            row = {"a": a, "b": b, "split": split, "_theseus_item_id": f"{split}-{i}"}
            if kind == "classify":
                row["label"] = "yes" if a + (b - 100) / 20 > 0 else "no"
            else:
                row["label"] = 3.0 * a + 0.05 * b + float(rng.normal(0, 0.1))
            if split == "train" and i % 17 == 0:
                row["b"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def train_table(tmp_path, monkeypatch, kind: str, **selection):
    df = make_table(np.random.default_rng(0), kind)
    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(
        train_mod.storage, "upload_dir", lambda b, p, local_dir, **kw: shutil.copytree(local_dir, uploaded)
    )
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    task_id = "tabular_classification" if kind == "classify" else "tabular_regression"
    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="a", kind="scalar"),
            ColumnSpec(name="b", kind="scalar"),
            ColumnSpec(name="label", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=["no", "yes"] if kind == "classify" else [],
    )
    config = compile_pytorch_config(
        get_task_descriptor(task_id), snapshot, PytorchHyperparameters(seed=1, epochs=40, batch_size=32, **selection)
    )
    reports: list = []
    ctx = TrainContext(
        run_id="smoke", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )  # fmt: skip
    model = PytorchBackend.train(ctx)
    return types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, config=config)


def load_client(model_path, meta_path):
    module = types.ModuleType("omnitrain_client_table_smoke")
    module.__file__ = str(CLIENT_PATH)
    exec(compile(CLIENT_PATH.read_text(encoding="utf-8"), str(CLIENT_PATH), "exec"), module.__dict__)  # noqa: S102
    return module.OmniTrainClient(model_path, meta_path)


def bundle_for(run, tmp_path) -> Path:
    """What an export bundle holds: the one-file model and the manifest built from names.json."""
    path = PytorchBackend.convert(PytorchBackend.load(str(run.uploaded)), "onnx", str(tmp_path / "export"))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copyfile(path, bundle / "model.onnx")
    meta = json.loads((run.uploaded / "names.json").read_text(encoding="utf-8"))
    (bundle / "preprocessing.json").write_text(json.dumps(build_preprocessing_manifest("smoke", {}, meta)))
    return bundle


def test_a_table_classifier_trains_scores_rows_with_gaps_and_exports(tmp_path, monkeypatch):
    import onnxruntime as ort
    import torch

    run = train_table(tmp_path, monkeypatch, "classify", learning_rate=0.01)
    assert sorted(p.name for p in run.uploaded.iterdir()) == ["model.pt", "names.json"]
    assert all("loss" in m and "accuracy" in m for _, _, m in run.reports)
    assert max(m["accuracy"] for _, s, m in run.reports if s == "validation") > 0.85

    # The statistics are the TRAIN rows' alone, though test holds values 50 units out.
    meta = json.loads((run.uploaded / "names.json").read_text(encoding="utf-8"))
    train_rows = run.df[run.df["split"] == "train"]
    assert meta["inputKind"] == "tabular" and meta["names"] == ["no", "yes"] and meta["inputColumns"] == ["a", "b"]
    assert meta["mean"] == pytest.approx([train_rows["a"].mean(), train_rows["b"].mean()], rel=1e-5)
    assert meta["std"] == pytest.approx([train_rows["a"].std(ddof=0), train_rows["b"].std(ddof=0)], rel=1e-4)
    assert meta["fill"] == pytest.approx(meta["mean"])

    loaded = PytorchBackend.load(str(run.uploaded))
    assert loaded.output.labels == ["no", "yes"] and loaded.input_columns == ["a", "b"]
    gap = pd.DataFrame({"a": [2.0, 0.0], "b": [np.nan, "not a number"]})
    frame = loaded.predict(gap)
    assert np.allclose(np.array(frame["label_probabilities"].tolist()).sum(axis=1), 1.0, atol=1e-4)
    assert loaded.to_output(frame)["kind"] == "classification"

    result = PytorchBackend.evaluate(loaded, run.df, "split", "_theseus_item_id")
    assert result is not None and result.report["split"] == "test" and result.report["rowCount"] == 40
    assert result.report["idx2str"] == ["no", "yes"]
    correct = np.trace(np.array(result.report["confusionMatrix"]))
    assert correct / 40 > 0.8

    # ONNX: raw columns in (one tensor per column, named by it), logits out, a gap or an infinity filled inside.
    bundle = bundle_for(run, tmp_path)
    session = ort.InferenceSession(str(bundle / "model.onnx"))
    assert [i.name for i in session.get_inputs()] == ["a", "b"] and session.get_outputs()[0].name == "label"
    raw = np.array([[1.5, 120.0], [np.nan, 90.0], [0.3, np.inf]], dtype=np.float32)
    with torch.no_grad():
        expected = loaded.net(torch.from_numpy(raw).to(runtime.device())).cpu().numpy()
    (logits,) = session.run(["label"], {"a": raw[:, 0], "b": raw[:, 1]})
    np.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-4)

    # The shipped client, with no normalisation of its own, agrees with the platform, including on a gap.
    client = load_client(bundle / "model.onnx", bundle / "preprocessing.json")
    assert client.manifest["inputs"][0].get("numberNormalization") is None
    for record in ({"a": 1.5, "b": 120.0}, {"a": float("nan"), "b": 90.0}):
        from_client = client.predict(record, top_k=2)
        from_platform = loaded.golden_prediction(loaded.predict(pd.DataFrame([record])))
        assert from_client.keys() == from_platform.keys()
        for label, p in from_client.items():
            assert p == pytest.approx(from_platform[label], abs=2e-3)


def test_a_table_regressor_trains_reports_in_the_targets_units_and_exports(tmp_path, monkeypatch):
    import onnxruntime as ort
    import torch

    run = train_table(tmp_path, monkeypatch, "regress", learning_rate=0.01)
    assert all("loss" in m and "mean_absolute_error" in m for _, _, m in run.reports)
    meta = json.loads((run.uploaded / "names.json").read_text(encoding="utf-8"))
    train_target = run.df[run.df["split"] == "train"]["label"]
    assert meta["names"] == [] and meta["targetMean"] == pytest.approx(train_target.mean(), rel=1e-4)
    # A reported loss is a squared error in the target's own units, so it is comparable with scikit-learn's.
    validation = [m for _, s, m in run.reports if s == "validation"]
    assert min(m["mean_absolute_error"] for m in validation) < 3.0

    loaded = PytorchBackend.load(str(run.uploaded))
    assert loaded.output.kind == "regression" and loaded.output.labels is None
    out = loaded.to_output(loaded.predict(pd.DataFrame({"a": [1.0], "b": [100.0]})))
    assert out["kind"] == "regression"

    result = PytorchBackend.evaluate(loaded, run.df, "split", "_theseus_item_id")
    assert result is not None and result.report["outputType"] == "number"
    assert result.report["overall"]["r2"] > 0.8
    assert "label_actual" in result.predictions.columns

    bundle = bundle_for(run, tmp_path)
    session = ort.InferenceSession(str(bundle / "model.onnx"))
    raw = np.array([[1.5, 120.0], [np.nan, 90.0]], dtype=np.float32)
    with torch.no_grad():
        expected = loaded.net(torch.from_numpy(raw).to(runtime.device())).cpu().numpy()
    (values,) = session.run(["label"], {"a": raw[:, 0], "b": raw[:, 1]})
    np.testing.assert_allclose(values.reshape(-1), expected, rtol=1e-4, atol=1e-3)

    client = load_client(bundle / "model.onnx", bundle / "preprocessing.json")
    record = {"a": 1.5, "b": 120.0}
    assert client.predict(record)["value"] == pytest.approx(
        float(loaded.predict(pd.DataFrame([record]))["label_predictions"].iloc[0]), abs=1e-3
    )


def test_class_weights_and_the_other_architectures_train(tmp_path, monkeypatch):
    run = train_table(tmp_path, monkeypatch, "classify", model_id="mlp_small", use_class_weights=True, dropout=0.0)
    assert run.config["train"]["useClassWeights"] is True and run.config["network"]["hidden"] == [64, 64]
    assert any(s == "validation" for _, s, _ in run.reports)


def test_a_train_split_with_one_class_is_refused(tmp_path, monkeypatch):
    from omnitrain.plugins.backends.base import ConfigError

    df = make_table(np.random.default_rng(0), "classify")
    df["label"] = "yes"
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="a", kind="scalar"),
            ColumnSpec(name="b", kind="scalar"),
            ColumnSpec(name="label", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=["no", "yes"],
    )
    config = compile_pytorch_config(
        get_task_descriptor("tabular_classification"), snapshot, PytorchHyperparameters(epochs=2)
    )
    ctx = TrainContext(
        run_id="x", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None, _report=lambda *a: None,
    )  # fmt: skip
    with pytest.raises(ConfigError, match="only one class"):
        PytorchBackend.train(ctx)
    assert not (tmp_path / "work").exists()


def test_mixed_precision_and_gradient_clipping_train_on_a_gpu(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("mixed precision only does anything on a CUDA device")
    run = train_table(tmp_path, monkeypatch, "classify", learning_rate=0.01, mixed_precision=True, grad_clip_norm=1.0)
    assert run.config["train"]["mixedPrecision"] is True
    validation = [m for _, s, m in run.reports if s == "validation"]
    assert all(np.isfinite(m["loss"]) for m in validation) and max(m["accuracy"] for m in validation) > 0.85


TONES = {"low": 300.0, "mid": 1000.0, "high": 3000.0}


def write_tone(path: Path, label: str, rng: np.random.Generator) -> None:
    """A noisy tone per class, at a random sample rate and length, so resampling and windowing are both exercised.
    Stereo for some, with the tone only on one channel."""
    rate = int(rng.choice([8000, 16000, 22050]))
    seconds = float(rng.uniform(1.0, 4.0))
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.5 * np.sin(2 * np.pi * TONES[label] * t + rng.uniform(0, 6.28)) + rng.normal(0, 0.05, len(t))
    samples = np.stack([tone, np.zeros_like(tone)], axis=1) if rng.random() < 0.3 else tone
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1 if pcm.ndim == 1 else 2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def train_sounds(tmp_path, monkeypatch, **selection):
    rng = np.random.default_rng(0)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for split, count in (("train", 10), ("validation", 4), ("test", 4)):
        for label in TONES:
            for i in range(count):
                item_id = f"{split}-{label}-{i}"
                write_tone(source / f"{item_id}.wav", label, rng)
                rows.append(
                    {
                        "audio_path": f"s3://theseus-datasets/pool/{item_id}.wav",
                        "class": label,
                        "split": split,
                        "_theseus_item_id": item_id,
                    }
                )
    # A recording that is not audio must become a skipped row, not a failed run.
    (source / "broken.wav").write_bytes(b"this is not a sound")
    rows.append(
        {
            "audio_path": "s3://theseus-datasets/pool/broken.wav",
            "class": "low",
            "split": "train",
            "_theseus_item_id": "b",
        }
    )
    df = pd.DataFrame(rows)

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(source / os.path.basename(key), local_path)
        return local_path

    uploaded = tmp_path / "uploaded"
    monkeypatch.setattr(data.storage, "download_file", fake_download)
    monkeypatch.setattr(
        train_mod.storage, "upload_dir", lambda b, p, local_dir, **kw: shutil.copytree(local_dir, uploaded)
    )
    monkeypatch.setattr(train_mod.pd, "read_parquet", lambda uri: df)
    monkeypatch.setattr(runtime, "cache_dir", lambda: tmp_path / "cache")
    snapshot = SnapshotContext(
        columns=[
            ColumnSpec(name="audio_path", kind="storage_uri"),
            ColumnSpec(name="class", kind="label"),
            ColumnSpec(name="split", kind="split"),
        ],
        label_class_names=list(TONES),
    )
    defaults = {
        "model_id": "audio_cnn", "epochs": 12, "batch_size": 8, "learning_rate": 0.003, "seed": 1,
        "sample_rate": 8000, "clip_seconds": 2.0, "n_mels": 40, "clips_per_audio": 1,
    }  # fmt: skip
    config = compile_pytorch_config(
        get_task_descriptor("audio_classification"), snapshot, PytorchHyperparameters(**{**defaults, **selection})
    )
    reports: list = []
    ctx = TrainContext(
        run_id="smoke", config=config, dataset_uri="unused", output_uri="unused", workdir=str(tmp_path / "work"),
        _heartbeat=lambda: None, _check_abort=lambda: None,
        _report=lambda epoch, split, metrics: reports.append((epoch, split, metrics)),
    )  # fmt: skip
    model = PytorchBackend.train(ctx)
    return types.SimpleNamespace(model=model, reports=reports, uploaded=uploaded, df=df, source=source, config=config)


def test_a_sound_classifier_trains_serves_evaluates_and_exports(tmp_path, monkeypatch):
    import onnxruntime as ort
    import torch

    from omnitrain.plugins.backends.pytorch import audio
    from omnitrain.plugins.backends.pytorch.spec import AudioSpec

    run = train_sounds(tmp_path, monkeypatch)
    assert not (tmp_path / "work").exists()
    assert sorted(p.name for p in run.uploaded.iterdir()) == ["model.pt", "names.json"]
    assert all("loss" in m and "accuracy" in m for _, _, m in run.reports)
    assert max(m["accuracy"] for _, s, m in run.reports if s == "validation") > 0.8

    meta = json.loads((run.uploaded / "names.json").read_text(encoding="utf-8"))
    assert meta["inputKind"] == "audio" and meta["names"] == list(TONES) and meta["audioColumn"] == "audio_path"
    spec = AudioSpec.from_meta(meta)
    assert (spec.sample_rate, spec.n_mels, spec.frames) == (8000, 40, 1 + (16000 - 1024) // 256)
    # The scaling was measured on the train recordings, so it is neither the placeholder nor degenerate.
    assert spec.std > 0 and (spec.mean, spec.std) != (0.0, 1.0)

    loaded = PytorchBackend.load(str(run.uploaded))
    assert loaded.output.labels == list(TONES) and loaded.input_columns == ["audio_path"]
    file = run.source / "test-high-0.wav"
    frame = loaded.predict(pd.DataFrame({"audio_path": [str(file)]}))
    probs = np.array(frame["class_probabilities"].iloc[0])
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert loaded.golden_prediction(frame) and loaded.to_output(frame)["kind"] == "classification"

    result = PytorchBackend.evaluate(loaded, run.df, "split", "_theseus_item_id")
    assert result is not None and result.report["split"] == "test" and result.report["rowCount"] == 12
    assert result.report["idx2str"] == list(TONES)
    assert np.trace(np.array(result.report["confusionMatrix"])) / 12 > 0.7

    # ONNX: the bare network on a standardised spectrogram, one file, dynamic batch.
    path = PytorchBackend.convert(loaded, "onnx", str(tmp_path / "export"))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    session = ort.InferenceSession(path)
    assert session.get_inputs()[0].name == "spectrograms"
    assert session.get_inputs()[0].shape[1:] == [1, 40, spec.frames]
    wave_ = audio.load_waveform(str(file), spec)
    window = audio.cut(wave_, audio.window_starts(len(wave_), spec.length, 1)[0], spec.length)
    batch = np.stack([audio.spectrogram(window, spec)] * 3)[:, None]
    (logits,) = session.run(["logits"], {"spectrograms": batch})
    assert logits.shape == (3, 3)
    # What a client following the manifest would compute equals what the platform predicts (one window per clip).
    onnx_probs = torch.softmax(torch.from_numpy(logits[0]), dim=0).numpy()
    np.testing.assert_allclose(onnx_probs, probs, atol=2e-3)

    recipe = build_preprocessing_manifest("smoke", {}, meta)["inputs"][0]["audioSpectrogram"]
    assert recipe["normalization"] == {"mean": spec.mean, "std": spec.std}

    # The shipped Python client, following that manifest alone, agrees with the platform on this recording.
    bundle = bundle_for(run, tmp_path)
    client = load_client(bundle / "model.onnx", bundle / "preprocessing.json")
    from_client = client.predict(str(file), top_k=3)
    from_platform = loaded.golden_prediction(frame)
    assert from_client.keys() == from_platform.keys()
    for label, p in from_client.items():
        assert p == pytest.approx(from_platform[label], abs=2e-3)


def test_scoring_averages_windows_over_a_long_recording(tmp_path, monkeypatch):
    run = train_sounds(tmp_path, monkeypatch, epochs=3, clips_per_audio=3)
    loaded = PytorchBackend.load(str(run.uploaded))
    # A recording much longer than the 2 s window: three windows, one probability row summing to one.
    long_file = tmp_path / "long.wav"
    rng = np.random.default_rng(5)
    t = np.arange(8000 * 9) / 8000
    pcm = ((0.5 * np.sin(2 * np.pi * 1000 * t) + rng.normal(0, 0.05, len(t))) * 32767).astype("<i2")
    with wave.open(str(long_file), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(pcm.tobytes())
    frame = loaded.predict(pd.DataFrame({"audio_path": [str(long_file), str(run.source / "test-low-1.wav")]}))
    assert np.array(frame["class_probabilities"].tolist()).sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)

    # The shipped client picks the same three windows and averages them the same way.
    bundle = bundle_for(run, tmp_path)
    client = load_client(bundle / "model.onnx", bundle / "preprocessing.json")
    for i, path in enumerate([long_file, run.source / "test-low-1.wav"]):
        from_client = client.predict(str(path), top_k=3)
        from_platform = loaded.golden_prediction(frame.iloc[[i]])
        for label, p in from_client.items():
            assert p == pytest.approx(from_platform[label], abs=2e-3)


@pytest.mark.parametrize("model_id", [m.id for m in tasks.AUDIO_MODELS])
def test_every_audio_model_takes_a_spectrogram_trains_a_step_and_exports(model_id, tmp_path):
    import onnxruntime as ort
    import torch

    from omnitrain.plugins.backends.pytorch import network
    from omnitrain.plugins.backends.pytorch.spec import AudioSpec

    spec = AudioSpec(sample_rate=8000, seconds=2.0, n_mels=40)
    net = network.build_audio(model_id, 4, pretrained=False)
    net.train()
    x = torch.randn(2, 1, spec.n_mels, spec.frames)
    loss = torch.nn.functional.cross_entropy(net(x), torch.tensor([0, 3]))
    loss.backward()
    assert loss.item() == loss.item()  # not NaN
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)

    meta = {
        "model": model_id, **spec.to_meta(), "audioColumn": "audio_path", "outputName": "class",
        "names": ["a", "b", "c", "d"], "task": "classify",
    }  # fmt: skip
    loaded = wrap(net, meta)
    path = PytorchBackend.convert(loaded, "onnx", str(tmp_path))
    assert [p.name for p in Path(path).parent.iterdir()] == ["model.onnx"]
    batch = torch.randn(3, 1, spec.n_mels, spec.frames)
    with torch.no_grad():
        expected = loaded.net(batch.to(runtime.device())).cpu().numpy()
    (logits,) = ort.InferenceSession(path).run(["logits"], {"spectrograms": batch.numpy()})
    np.testing.assert_allclose(logits, expected, rtol=1e-2, atol=1e-2)


def test_freezing_an_audio_backbone_trains_only_the_head():
    from omnitrain.plugins.backends.pytorch import network

    net = network.build_audio("audio_resnet18", 3, pretrained=False, freeze_backbone=True)
    assert {n for n, p in net.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
    net = network.build_audio("audio_cnn", 3, pretrained=False, freeze_backbone=True)
    assert {n for n, p in net.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
