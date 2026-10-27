"""The preprocessing.json of a language-model export (pure, no torch).

There is no graph to describe: the model is a Hugging Face folder that carries its own tokenizer. What a client must
repeat is the platform's recipe around it, which is this manifest's `generation` block: how the input fields become a
prompt, how a long prompt is cut, and how the answer is decoded. `prompt.py` and `data.py` are the reference
implementation, and the generated Python example follows this object (and refuses a `recipe` it does not know).
"""

from typing import Any

from omnitrain.plugins.backends.llm import prompt
from omnitrain.plugins.backends.llm.runtime import METADATA_FILENAME

RECIPE = "causal_lm_prompt_v1"

PROMPT_LAYOUT = (
    "For each input field in inputColumns, in that order, a section '### <name>:\\n<value>\\n' (the value is the "
    "text with surrounding whitespace removed, empty when it is missing). Sections are joined with a newline, and the "
    "prompt ends with '\\n### <outputColumn>:\\n'."
)
PROMPT_CUT = (
    "Tokenise the prompt with the model's tokenizer, special tokens included. When it has more than promptRoom ids, "
    "keep the last promptRoom of them, except that a leading beginning-of-text id stays first (then the last "
    "promptRoom - 1 follow it)."
)
DECODING = (
    "Greedy: always the most likely next token, no sampling, until the end-of-text token or maxNewTokens new tokens. "
    "The answer is the new tokens decoded without special tokens, with surrounding whitespace removed."
)


def build_preprocessing_manifest(run_id: str, config: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    if meta is None:
        raise ValueError(f"Run {run_id} has no {METADATA_FILENAME} to build a preprocessing manifest from")
    text = meta["text"]
    inputs = [{"name": column, "type": "text", "column": column} for column in meta["inputColumns"]]
    output = {"name": meta["outputColumn"], "type": "text", "column": meta["outputColumn"]}
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "inputs": inputs,
        "outputs": [output],
        "generation": {
            "recipe": RECIPE,
            "inputColumns": list(meta["inputColumns"]),
            "outputColumn": meta["outputColumn"],
            "maxLength": int(text["maxLength"]),
            "maxNewTokens": int(text["maxNewTokens"]),
            "promptRoom": prompt.prompt_room(int(text["maxLength"]), int(text["maxNewTokens"])),
            "promptLayout": PROMPT_LAYOUT,
            "promptCut": PROMPT_CUT,
            "decoding": DECODING,
        },
    }
