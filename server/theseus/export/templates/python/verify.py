"""Checks that this client's own preprocessing reproduces the platform's
prediction for the bundled sample. `theseus_client.py` re-implements
Ludwig's preprocessing from preprocessing.json; nothing else in this bundle
proves that reimplementation is correct — this script is that proof.

Exits non-zero (and prints a diff) if the top-1 class or any shared
probability disagrees by more than TOLERANCE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from theseus_client import TheseusClient

TOLERANCE = 1e-3
_HERE = Path(__file__).parent


def main() -> int:
    expected_path = _HERE / "expected.json"
    if not expected_path.exists():
        print("No expected.json in this bundle — nothing to verify against.")
        return 0

    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)

    sample_path = _HERE / expected["sampleFile"]
    client = TheseusClient()
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
