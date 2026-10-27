"""Minimal usage example.

A model with one input field takes the text itself; one with several takes a JSON object of them:

    python example.py "Translate this sentence."
    python example.py '{"question": "Who wrote it?", "context": "It was written by ..."}'
"""

import json
import sys

from omnitrain_llm import OmniTrainLLM

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <text | JSON object of field values>")
        sys.exit(1)

    model = OmniTrainLLM()
    if len(model.input_columns) > 1:
        try:
            inputs = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            print(f"This model takes several fields; pass a JSON object with {model.input_columns}")
            sys.exit(1)
    else:
        inputs = sys.argv[1]
    print(model.predict(inputs))
