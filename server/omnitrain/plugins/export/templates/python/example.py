"""Minimal usage example.

Run with an image path for an image-input model, or a JSON object (one
value per input column) for a tabular model — e.g.:

    python example.py cat.jpg
    python example.py '{"age": 34, "income": 52000}'

An object detector prints one line per box; a video model takes a video file path and an audio
model an audio file path (a WAV file; other formats need `pip install soundfile`). A text model takes
the text itself: `python example.py "what a great film"`. A tagger takes the text too and prints one line per
word: the word, its tag and the model's confidence.
"""

import json
import sys

from omnitrain_client import OmniTrainClient

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-image | JSON object of column values>")
        sys.exit(1)

    client = OmniTrainClient()

    if client._is_tabular:  # noqa: SLF001 — example code, not a library boundary
        try:
            record = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            print("This model takes tabular input — pass a JSON object, e.g. '{\"age\": 34}'")
            sys.exit(1)
        predictions = client.predict(record)
    else:
        predictions = client.predict(sys.argv[1])

    if isinstance(predictions, list) and client._kind == "tags":  # noqa: SLF001 — `[{token, tag, confidence}]`
        for word in predictions:
            print(f"{word['token']}\t{word['tag']}\t{word['confidence']:.4f}")
    elif isinstance(predictions, list):  # an object detector: `[{label, confidence, box}]`
        for detection in predictions:
            box = detection["box"]
            print(
                f"{detection['label']}: {detection['confidence']:.4f}  "
                f"x={box['x']:.3f} y={box['y']:.3f} w={box['w']:.3f} h={box['h']:.3f}"
            )
        if not predictions:
            print("No objects found.")
    else:
        for class_name, probability in predictions.items():
            print(f"{class_name}: {probability:.4f}")
