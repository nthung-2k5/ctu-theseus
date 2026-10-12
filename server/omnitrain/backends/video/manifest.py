"""The preprocessing.json that makes a bare video ONNX export usable.

The exported graph takes one ready-made clip tensor and returns raw class logits: no decoding of a video file, no
frame sampling, no resize, no label lookup. `names.json` (written at train time) supplies what a client must do to
a video first and the class list in the model's own index order.

The class list MUST come from `names.json`, NEVER from the label_classes table: Postgres has no idea which index a
run assigned each class, so guessing wrong silently mislabels every prediction.

No generated devkit or app implements video yet (see export/formats/_kits.py), so a video bundle is the bare model
and this manifest is the contract a hand-written client follows.
"""

from typing import Any

from omnitrain.backends.video.runtime import METADATA_FILENAME


def build_preprocessing_manifest(run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: turn the training metadata into the preprocessing manifest."""
    if meta is None:
        raise ValueError(f"Run {run_id} has no {METADATA_FILENAME} to build a preprocessing manifest from")

    size, frames, stride = meta["frameSize"], meta["numFrames"], meta["frameStride"]
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "inputs": [
            {
                "name": "clips",
                "type": "video",
                "column": meta["clipColumn"],
                "layout": "NCTHW",
                # How a video file becomes the input tensor (omnitrain/backends/video/frames.py): take `clipsPerVideo`
                # clips of `numFrames` frames, `frameStride` source frames apart, with starts spread evenly over the
                # video (one centred clip if there is one), repeating the last frame of a video that is too short.
                # Score each clip, apply a softmax to the logits, and average the probabilities across clips.
                "videoSampling": {
                    "numFrames": frames,
                    "frameStride": stride,
                    "clipsPerVideo": meta["clipsPerVideo"],
                    "clipStarts": "evenly_spaced",
                    "shortVideo": "repeat_last_frame",
                },
                # Per frame: RGB, resize the SHORTER edge to `size` (bilinear), centre-crop a size x size square.
                "imageResize": {"mode": "shorter_edge_center_crop", "interpolation": "bilinear"},
                # Scaled to 0..1 first, then (x - mean) / std per channel.
                "imageNormalization": {"mean": list(meta["mean"]), "std": list(meta["std"])},
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
