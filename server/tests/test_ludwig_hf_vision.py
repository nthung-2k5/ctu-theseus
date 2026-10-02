"""The `hf_vision` encoder through REAL Ludwig: a tiny random backbone is trained, saved, reloaded and used
to predict. Everything else about custom models is tested with Ludwig faked; this is the one place that proves
Ludwig accepts the encoder's config, runs it, and rebuilds it from a saved model (where the weights come from
Ludwig's checkpoint and only config.json is read from the model folder)."""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
pytest.importorskip("ludwig")

from PIL import Image  # noqa: E402

from theseus import constants as C  # noqa: E402
from theseus.backends.base import CustomModelRef  # noqa: E402
from theseus.backends.ludwig import model as ludwig_model  # noqa: E402  (registers the encoder)
from theseus.backends.ludwig.compile import LudwigHyperparameters, compile_ludwig_config  # noqa: E402
from theseus.services.storage import find_model_dir  # noqa: E402
from theseus.services.task_registry import SnapshotContext, get_task_descriptor  # noqa: E402


def tiny_backbone(kind: str, root):
    from transformers import ConvNextConfig, ConvNextModel, ViTConfig, ViTModel

    if kind == "vit":  # a transformer: pooled from tokens
        model = ViTModel(
            ViTConfig(
                image_size=32,
                patch_size=8,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_channels=3,
            )  # fmt: skip
        )
    else:  # a conv net: pooled from a feature map, and with no image_size in its config
        model = ConvNextModel(ConvNextConfig(num_channels=3, num_stages=2, hidden_sizes=[8, 16], depths=[1, 1]))
    model.save_pretrained(root)
    return root


def dataset(tmp_path) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        red = i % 2 == 0
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[..., 0 if red else 2] = 200
        pixels = np.clip(pixels + rng.integers(0, 40, pixels.shape), 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(tmp_path / f"img{i}.png")
        rows.append(
            {
                C.IMAGE_PATH_COLUMN_NAME: f"img{i}.png",  # relative: Ludwig misreads a Windows "C:\\..." path
                C.CLASS_COLUMN_NAME: "red" if red else "blue",
                C.SPLIT_INDEX_COLUMN_NAME: 0 if i < 16 else (1 if i < 20 else 2),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("backbone", ["vit", "convnext"])
def test_a_custom_vision_backbone_trains_saves_reloads_and_predicts(backbone, tmp_path, monkeypatch):
    from ludwig.api import LudwigModel

    # Ludwig's model.save() calls os.fsync on a descriptor Windows rejects (EBADF), for ANY model. It is a
    # durability nicety, irrelevant here, and a no-op on the Linux the service runs on.
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    monkeypatch.chdir(tmp_path)
    folder = tiny_backbone(backbone, tmp_path / "hf")
    df = dataset(tmp_path)

    custom = CustomModelRef(id="custom:x", kind="hf_vision", source_kind="upload", local_path=str(folder))
    config = compile_ludwig_config(
        get_task_descriptor("image_classification"),
        SnapshotContext(label_class_names=["red", "blue"]),
        LudwigHyperparameters(encoderId="custom:x", epochs=2, batchSize=4, learningRate=0.01),
        custom,
    )
    assert config["input_features"][0]["encoder"]["type"] == "hf_vision"

    trained = LudwigModel(config=config, logging_level=30)
    trained.train(
        dataset=df, output_directory=str(tmp_path / "out"), experiment_name="results", skip_save_processed_input=True
    )
    predictions, _ = trained.predict(dataset=df, skip_save_predictions=True)
    column = f"{C.CLASS_COLUMN_NAME}_predictions"
    assert set(predictions[column]) <= {"red", "blue"} and len(predictions) == len(df)

    # Reload the way the API does after a restart: from the saved directory, with only config.json to read.
    reloaded = LudwigModel.load(find_model_dir(str(tmp_path / "out")))
    again, _ = reloaded.predict(dataset=df, skip_save_predictions=True)
    assert (again[column].values == predictions[column].values).all()
    assert ludwig_model is not None
