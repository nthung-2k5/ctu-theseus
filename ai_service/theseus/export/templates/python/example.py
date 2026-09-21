"""Minimal usage example.

Run with an image path for an image-input model, or a JSON object (one
value per input column) for a tabular model — e.g.:

    python example.py cat.jpg
    python example.py '{"age": 34, "income": 52000}'
"""

import json
import sys

from theseus_client import TheseusClient

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-image | JSON object of column values>")
        sys.exit(1)

    client = TheseusClient()

    if client._is_tabular:  # noqa: SLF001 — example code, not a library boundary
        try:
            record = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            print("This model takes tabular input — pass a JSON object, e.g. '{\"age\": 34}'")
            sys.exit(1)
        predictions = client.predict(record)
    else:
        predictions = client.predict(sys.argv[1])

    for class_name, probability in predictions.items():
        print(f"{class_name}: {probability:.4f}")
