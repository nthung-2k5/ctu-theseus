"""The one export artifact of a language model: its Hugging Face folder, zipped. Imports torch.

A bundle carries a trained model as the same self-contained folder the platform serves it from (weights, tokenizer,
`names.json`). The zip stores its files uncompressed: weights barely compress, and a bundle is unzipped again when it is
assembled, so deflating them twice would only cost time.
"""

import zipfile
from pathlib import Path

from omnitrain.plugins.backends.base import ARTIFACT_FILENAMES, Artifact
from omnitrain.plugins.backends.llm.model import LlmLoadedModel

ARTIFACTS: dict[str, Artifact] = {"hf_model": Artifact("hf_model", ARTIFACT_FILENAMES["hf_model"])}

# A bundle is assembled in memory, the model held about three times over (downloaded, unzipped, zipped again).
MAX_EXPORT_BYTES = 3 * 1024**3


def export_hf_model(model: LlmLoadedModel, workdir: str) -> str:
    """The model's folder as `model.zip` under `workdir`; its path."""
    size = model.weight_bytes()
    if size > MAX_EXPORT_BYTES:
        raise ValueError(
            f"This model's weights take {size / 1024**3:.1f} GiB, more than the {MAX_EXPORT_BYTES // 1024**3} GiB a "
            "bundle can hold."
        )
    folder = Path(workdir) / "hf_model"
    model.save(str(folder))
    target = Path(workdir) / ARTIFACTS["hf_model"].filename
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())
    return str(target)
