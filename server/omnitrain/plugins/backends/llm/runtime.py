"""The device and number format a language model runs in. Imports torch only inside functions."""

from typing import Any

# Files a trained run uploads under its results prefix (and `load` reads back from the downloaded copy), beside the
# merged model's own config.json / model.safetensors and the tokenizer files.
METADATA_FILENAME = "names.json"


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype() -> Any:
    """A `torch.dtype`: bfloat16 on a GPU that has it (half the memory, and unlike float16 it needs no loss scaling),
    else float32: a CPU gains nothing from half precision, and some CPUs have no kernels for it."""
    import torch

    # Not `is_bf16_supported()` alone: by default it also says yes to a GPU that only emulates bfloat16 (a Turing
    # card such as the GTX 1650), which is far slower than float32.
    native = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.bfloat16 if native else torch.float32
