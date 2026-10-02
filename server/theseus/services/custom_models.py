"""Bring-your-own models: where their files live locally, putting them there, and the generic file checks.

A custom model's weights are either on the Hugging Face Hub (fetched at a pinned commit) or an uploaded
bundle in `theseus-models/custom/{modelId}/`. A trainer backend reads them from a local directory whose
path is written into the compiled config, so that path is DETERMINISTIC (model id + pinned revision or
content hash) and `ensure_local` recreates it on demand: before a run trains, and again before a trained
model is loaded for inference or export, even after a restart wiped the temp directory.

Everything here is synchronous (boto3, huggingface_hub, zipfile): call it through an executor.

What this module refuses, whatever the framework, because the files came from a user or a Hub author:
  * pickle-based weights and anything executable (see `FORBIDDEN_EXTENSIONS`): loading those runs code;
  * symlinks and paths that climb out of the model directory (zip-slip);
  * a bundle bigger than `max_custom_model_bytes`, before or after decompression (zip bombs).
Framework-specific checks (is this really a causal LM?) belong to `TrainerBackend.validate_custom_model`.
"""

import hashlib
import logging
import os
import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from theseus import constants as C
from theseus.backends.base import CustomModelRef
from theseus.db.models import CustomModel
from theseus.services import storage
from theseus.settings import get_settings

logger = logging.getLogger(__name__)

# Written only after materialisation finishes, so a half-downloaded directory is never mistaken for a hit
# (same convention as storage.download_model).
_COMPLETE_MARKER = ".materialized"

BUNDLE_ZIP = "bundle.zip"
BUNDLE_SAFETENSORS = "model.safetensors"

# Loading any of these can execute arbitrary code (pickle) or is code outright.
FORBIDDEN_EXTENSIONS = frozenset(
    {
        ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".h5", ".hdf5", ".msgpack",
        ".py", ".pyc", ".pyd", ".so", ".dll", ".dylib", ".exe", ".sh", ".bat", ".cmd", ".ps1", ".jar",
    }
)  # fmt: skip

# What a Hub snapshot may fetch: weights as safetensors plus the small config/tokenizer text files.
_HUB_ALLOW = ("*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.md")


class CustomModelFileError(ValueError):
    """The model's files are not acceptable; the message is safe to show the user."""


def custom_model_key(model_id: uuid.UUID | str, filename: str) -> str:
    """The `theseus-models` key of an uploaded bundle."""
    return f"custom/{model_id}/{filename}"


def local_dir_for(row: CustomModel) -> Path:
    """Deterministic: the same model, pinned the same way, is always at the same path."""
    pin = row.revision or row.sha256 or "unpinned"
    return get_settings().temp_dir / "custom_models" / str(row.id) / pin


def ref_for(row: CustomModel) -> CustomModelRef:
    return CustomModelRef(
        id=f"custom:{row.id}",
        kind=row.kind,
        source_kind=row.source_kind,  # type: ignore[arg-type]
        source_ref=row.source_ref,
        revision=row.revision,
        local_path=str(local_dir_for(row)),
        spec=dict(row.spec or {}),
    )


def needs_files(row: CustomModel) -> bool:
    """False for a kind whose framework fetches its own weights by name (`CustomModelKind.materialize`)."""
    return bool((row.spec or {}).get("materialize", True))


def ensure_local(row: CustomModel) -> Path:
    """Make the model's files exist at `local_dir_for(row)` and return that path. Idempotent."""
    dest = local_dir_for(row)
    if not needs_files(row):
        return dest
    if (dest / _COMPLETE_MARKER).is_file():
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    if row.source_kind == "hub":
        _fetch_hub(row, dest)
    else:
        _fetch_upload(row, dest)
    (dest / _COMPLETE_MARKER).touch()
    return dest


def _fetch_hub(row: CustomModel, dest: Path) -> None:
    if not row.source_ref or not row.revision:
        raise CustomModelFileError("This Hub model has not been validated yet, so it has no pinned revision")
    from huggingface_hub import snapshot_download  # lazy: only Hub models need it

    snapshot_download(
        repo_id=row.source_ref,
        revision=row.revision,
        local_dir=str(dest),
        allow_patterns=list(_HUB_ALLOW),
        ignore_patterns=[f"*{ext}" for ext in sorted(FORBIDDEN_EXTENSIONS)],
    )


def _fetch_upload(row: CustomModel, dest: Path) -> None:
    """Re-materialise an already-validated upload at its pinned path, checking it is still the same bytes."""
    sha256 = _download_and_extract(row, dest)
    if row.sha256 and sha256 != row.sha256:
        raise CustomModelFileError("The stored bundle no longer matches the checksum recorded when it was validated")


@dataclass(frozen=True)
class ImportedUpload:
    sha256: str
    size_bytes: int
    path: Path


def import_upload(row: CustomModel) -> ImportedUpload:
    """First contact with an uploaded bundle (the validation job): download it once, hash it, and unpack it
    into the directory that hash names, so `ensure_local` finds it ready afterwards.

    The pin (and therefore the path) is the bundle's own sha256, which is not known until it has been
    read, so it is unpacked beside its final home and moved into place.
    """
    root = get_settings().temp_dir / "custom_models" / str(row.id)
    staging = root / f".staging-{uuid.uuid4().hex}"
    try:
        sha256 = _download_and_extract(row, staging)
        final = root / sha256
        shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
        (final / _COMPLETE_MARKER).touch()
        size = sum(p.stat().st_size for p in final.rglob("*") if p.is_file() and p.name != _COMPLETE_MARKER)
        return ImportedUpload(sha256=sha256, size_bytes=size, path=final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _download_and_extract(row: CustomModel, dest: Path) -> str:
    """Download the uploaded object, extract (or place) it in `dest`, and return the bundle's sha256."""
    if not row.storage_key:
        raise CustomModelFileError("This model has no uploaded file")
    name = PurePosixPath(row.storage_key).name
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / f".{uuid.uuid4().hex}.{name}.part"
    try:
        storage.download_file(C.BUCKET_MODELS, row.storage_key, str(archive))
        sha256 = sha256_file(archive)
        if name.endswith(".zip"):
            extract_bundle(archive, dest, max_bytes=get_settings().max_custom_model_bytes)
        else:
            shutil.move(str(archive), str(dest / name))
        return sha256
    finally:
        archive.unlink(missing_ok=True)


# -- Extraction and checks -----------------------------------------------------------------------


def extract_bundle(archive: Path, dest: Path, *, max_bytes: int) -> None:
    """Unzip a model bundle into `dest`, refusing anything unsafe. A single top-level folder is stripped,
    so a zip of `my-model/…` and a zip of the files themselves lay out the same way."""
    root = dest.resolve()
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise CustomModelFileError("The upload is not a valid zip file") from None
    with zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        if not members:
            raise CustomModelFileError("The zip file is empty")
        if sum(i.file_size for i in members) > max_bytes:
            raise CustomModelFileError(f"The model is larger than the {max_bytes // 2**20} MiB limit once unzipped")

        parts = [PurePosixPath(i.filename).parts for i in members]
        strip = 1 if len({p[0] for p in parts}) == 1 and all(len(p) > 1 for p in parts) else 0

        for info, p in zip(members, parts, strict=True):
            if stat.S_ISLNK(info.external_attr >> 16):
                raise CustomModelFileError(f"'{info.filename}' is a symbolic link, which is not allowed")
            rel = PurePosixPath(*p[strip:])
            if rel.is_absolute() or ".." in rel.parts:
                raise CustomModelFileError(f"'{info.filename}' points outside the model folder")
            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                raise CustomModelFileError(f"'{info.filename}' points outside the model folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def check_tree(root: Path, *, max_bytes: int) -> int:
    """Generic safety check of a materialised model directory. Returns its total size in bytes."""
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name == _COMPLETE_MARKER:
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                raise CustomModelFileError(f"'{path.relative_to(root)}' is a symbolic link, which is not allowed")
            ext = path.suffix.lower()
            if ext in FORBIDDEN_EXTENSIONS:
                raise CustomModelFileError(
                    f"'{path.relative_to(root)}' is a {ext} file. Pickle-based weights and code are not accepted; "
                    "convert the weights to .safetensors"
                )
            total += path.stat().st_size
    if total > max_bytes:
        raise CustomModelFileError(f"The model is {total // 2**20} MiB, over the {max_bytes // 2**20} MiB limit")
    if total == 0:
        raise CustomModelFileError("The model has no files")
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
