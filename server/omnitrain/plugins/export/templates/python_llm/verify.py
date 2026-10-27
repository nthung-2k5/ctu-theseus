"""Checks that this client writes what the platform wrote for the bundled sample.

`omnitrain_llm.py` repeats the platform's prompt layout, prompt cut and greedy decoding around the Hugging Face model in
`model/`; nothing else in this bundle proves that repetition is right, and this script is that proof. It exits non-zero
(and prints both answers) when the two texts differ.

The answers are compared exactly. Greedy decoding picks the single most likely token each step, so a different number
format (a GPU's bfloat16 against a CPU's float32) can, rarely, pick another near-tied token and write another answer.
"""

import json
import sys
from pathlib import Path

from omnitrain_llm import OmniTrainLLM

_HERE = Path(__file__).parent


def main() -> int:
    expected_path = _HERE / "expected.json"
    if not expected_path.exists():
        print("No expected.json in this bundle — nothing to verify against.")
        return 0
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected.get("text") is None:
        print("expected.json has no text to compare against.")
        return 0

    sample = (_HERE / expected["sampleFile"]).read_text(encoding="utf-8")
    model = OmniTrainLLM()
    inputs = json.loads(sample) if len(model.input_columns) > 1 else sample
    found = model.predict(inputs)
    if found != expected["text"]:
        print("MISMATCH the platform wrote:")
        print(f"  {expected['text']!r}")
        print("but this client wrote:")
        print(f"  {found!r}")
        return 1
    print(f"OK — the answer matches the platform's output ({len(found)} characters).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
