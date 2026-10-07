"""The preprocessing.json that makes a bare YOLO ONNX export usable.

The exported graph takes a ready-made float tensor and returns raw class probabilities: no decoding of an
image file, no resize, no label lookup. `names.json` (written at train time from the trained model itself)
supplies both what a client must do to an image first and the class list in the model's own index order.

The class list MUST come from `names.json`, NEVER from the label_classes table: Postgres has no idea which
index a run assigned each class, so guessing wrong silently mislabels every prediction a generated client makes.
"""

from typing import Any

from theseus.backends.yolo.runtime import METADATA_FILENAME  # noqa: F401  re-exported for the backend class


def build_preprocessing_manifest(run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: turn the training-set metadata into the preprocessing manifest."""
    if meta is None:
        raise ValueError(f"Run {run_id} has no {METADATA_FILENAME} to build a preprocessing manifest from")

    size = meta["imgsz"]
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "inputs": [
            {
                "name": "images",
                "type": "image",
                "column": meta["imageColumn"],
                # What Ultralytics' own inference does to an image (ultralytics.data.augment.classify_transforms):
                # resize the SHORTER edge to `size` (bilinear, keeping the aspect ratio), then centre-crop a
                # size x size square. Colour is RGB, scaled to 0..1, and its mean/std are 0/1: no further
                # normalisation. Every generated client implements this mode; without an `imageResize` they squash
                # the whole image to height x width, which is what Ludwig does and would distort a non-square image.
                "imageResize": {"mode": "shorter_edge_center_crop", "interpolation": "bilinear"},
                "imageNormalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
                # The key the generated clients read the target size from (they fall back to 224 without it).
                "ludwigPreprocessing": {"height": size, "width": size},
            }
        ],
        "outputs": [
            {
                "name": meta["outputName"],
                "type": "category",
                "column": meta["outputName"],
                "classes": list(meta["names"]),
            }
        ],
    }
