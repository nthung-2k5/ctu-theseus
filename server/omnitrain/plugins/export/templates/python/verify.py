"""Checks that this client's own preprocessing reproduces the platform's
prediction for the bundled sample. `omnitrain_client.py` re-implements
Ludwig's preprocessing from preprocessing.json; nothing else in this bundle
proves that reimplementation is correct — this script is that proof.

Exits non-zero (and prints a diff) if the top-1 class or any shared
probability disagrees by more than TOLERANCE.

For an object detector the comparison is of boxes, not probabilities: see
`verify_detections`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omnitrain_client import OmniTrainClient

TOLERANCE = 1e-3
_HERE = Path(__file__).parent

# Object detection. The platform's prediction comes from the original weights through Ultralytics, which pads an
# image only up to a multiple of 32 before the network; this client pads to the full square the exported graph was
# built for. The same box therefore comes out a little different, and a borderline one may appear in only one list.
# So only confident boxes are held to account, and by overlap and a loose confidence margin, not exact numbers.
CONFIDENT = 0.35  # a box at least this confident on either side must have a partner on the other
MIN_IOU = 0.7
CONFIDENCE_MARGIN = 0.1


def _iou(a: dict, b: dict) -> float:
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    width = min(ax2, bx2) - max(a["x"], b["x"])
    height = min(ay2, by2) - max(a["y"], b["y"])
    intersection = max(width, 0.0) * max(height, 0.0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def _partner(box: dict, others: list[dict]) -> dict | None:
    """The best-overlapping box of the same class in `others`, if it overlaps enough and is about as confident."""
    best = max(
        (o for o in others if o["label"] == box["label"]),
        key=lambda o: _iou(box["box"], o["box"]),
        default=None,
    )
    if best is None or _iou(box["box"], best["box"]) < MIN_IOU:
        return None
    return best if abs(best["confidence"] - box["confidence"]) <= CONFIDENCE_MARGIN else None


def verify_detections(expected: list[dict], actual: list[dict]) -> bool:
    ok = True
    for source, side, other, other_name in (
        (expected, "platform", actual, "this client"),
        (actual, "this client", expected, "the platform"),
    ):
        for box in source:
            if box["confidence"] >= CONFIDENT and _partner(box, other) is None:
                ok = False
                b = box["box"]
                print(
                    f"MISMATCH {side} found '{box['label']}' ({box['confidence']:.2f}) at "
                    f"x={b['x']:.3f} y={b['y']:.3f} w={b['w']:.3f} h={b['h']:.3f}; {other_name} has no matching box"
                )
    if ok:
        confident = sum(1 for box in expected if box["confidence"] >= CONFIDENT)
        print(f"OK — {confident} confident box(es) match the platform's output (IoU >= {MIN_IOU}).")
    return ok


def main() -> int:
    expected_path = _HERE / "expected.json"
    if not expected_path.exists():
        print("No expected.json in this bundle — nothing to verify against.")
        return 0

    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)

    sample_path = _HERE / expected["sampleFile"]
    client = OmniTrainClient()
    if client._kind == "audio" and sample_path.suffix.lower() != ".wav":  # noqa: SLF001 — verify script
        try:
            import soundfile  # noqa: F401
        except ImportError:
            print("The bundled sample is not a WAV file and `soundfile` is not installed — nothing to verify.")
            return 0
    if client._kind == "detection":  # noqa: SLF001 — verify script, not a library boundary
        if expected.get("detections") is None:
            print("expected.json has no detections to compare against.")
            return 0
        return 0 if verify_detections(expected["detections"], client.detect(sample_path)) else 1
    if client._is_tabular:  # noqa: SLF001 — verify script, not a library boundary
        with open(sample_path, encoding="utf-8") as f:
            sample_input = json.load(f)
    else:
        sample_input = sample_path
    actual = client.predict(sample_input, top_k=len(expected["predictions"]) or 5)

    expected_predictions: dict[str, float] = expected["predictions"]
    if not expected_predictions:
        print("expected.json has no predictions to compare against.")
        return 0

    expected_top1 = max(expected_predictions, key=lambda k: expected_predictions[k])
    actual_top1 = max(actual, key=lambda k: actual[k]) if actual else None

    ok = True
    if actual_top1 != expected_top1:
        ok = False
        print(f"MISMATCH top-1 class: expected '{expected_top1}', got '{actual_top1}'")

    for class_name, expected_prob in expected_predictions.items():
        actual_prob = actual.get(class_name)
        if actual_prob is None:
            ok = False
            print(f"MISMATCH class '{class_name}' missing from this client's output")
            continue
        diff = abs(actual_prob - expected_prob)
        if diff > TOLERANCE:
            ok = False
            print(f"MISMATCH class '{class_name}': expected {expected_prob:.4f}, got {actual_prob:.4f} (diff {diff:.4f})")

    if ok:
        print(f"OK — top-1 '{actual_top1}' matches platform output within {TOLERANCE}.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
