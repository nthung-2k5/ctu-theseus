"""Bring-your-own models: where their files live locally, putting them there, and the generic file checks.

A custom model's weights are either on the Hugging Face Hub (fetched at a pinned commit) or an uploaded
bundle in `theseus-models/custom/{modelId}/`. A Hub model may be private or gated: it is then fetched with
the OWNER's saved Hugging Face token (`services/integrations.py`), which callers pass in as `token`. With no
token the Hub is asked anonymously, never with whatever token happens to be configured on the server.

A trainer backend reads the files from a local directory whose path is written into the compiled config, so
that path is DETERMINISTIC (model id + pinned revision or content hash) and `ensure_local` recreates it on
demand: before a run trains, and again before a trained model is loaded for inference or export, even after
a restart wiped the temp directory.

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

# Loading any of these can execute arbitrary code (pickle) or is code outright.
FORBIDDEN_EXTENSIONS = frozenset(
    {
        ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".h5", ".hdf5", ".msgpack",
        ".py", ".pyc", ".pyd", ".so", ".dll", ".dylib", ".exe", ".sh", ".bat", ".cmd", ".ps1", ".jar",
    }
)  # fmt: skip

# What a Hub snapshot may fetch: weights as safetensors plus the small config/tokenizer text files.
_HUB_ALLOW = ("*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.md")
_MAX_CONFIG_BYTES = 2 * 2**20


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


def ensure_local(row: CustomModel, token: str | None = None) -> Path:
    """Make the model's files exist at `local_dir_for(row)` and return that path. Idempotent.

    `token` is the owner's Hugging Face token, needed only when the model is private or gated."""
    dest = local_dir_for(row)
    if not needs_files(row):
        return dest
    if (dest / _COMPLETE_MARKER).is_file():
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    if row.source_kind == "hub":
        _fetch_hub(row, dest, token)
    else:
        _fetch_upload(row, dest)
    (dest / _COMPLETE_MARKER).touch()
    return dest


@dataclass(frozen=True)
class HubPin:
    revision: str
    size_bytes: int
    has_safetensors: bool
    has_config: bool
    # Needs the owner's token to read: private, or gated and granted to them.
    private: bool = False


def _hub_token(token: str | None) -> str | bool:
    """What to hand huggingface_hub: the given token, or `False`, meaning NO token. `None` would make the
    library fall back to a token in the server's environment or cache, which must never speak for a user."""
    return token or False


def _access_error(repo_id: str, exc: BaseException, token: str | None) -> CustomModelFileError | None:
    """The user-facing reason Hugging Face refused to show or serve `repo_id`, or None for an unrelated error."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, GatedRepoError):
        if token:
            return CustomModelFileError(
                f"'{repo_id}' is gated and your Hugging Face account has not been granted access. "
                "Accept the model's terms on huggingface.co, then try again"
            )
        return CustomModelFileError(
            f"'{repo_id}' is gated. Connect your Hugging Face account in Settings (and accept the model's terms "
            "on huggingface.co) to import it"
        )
    if isinstance(exc, RepositoryNotFoundError):
        if token:
            return CustomModelFileError(f"'{repo_id}' was not found, or your Hugging Face account cannot read it")
        return CustomModelFileError(
            f"'{repo_id}' was not found on the Hugging Face Hub. If it is private, connect your Hugging Face "
            "account in Settings"
        )
    if isinstance(exc, HfHubHTTPError) and status in (401, 403):
        if token:
            return CustomModelFileError(
                "Hugging Face refused your saved token. Reconnect your Hugging Face account in Settings"
            )
        return CustomModelFileError(f"'{repo_id}' needs a Hugging Face account. Connect yours in Settings to import it")
    return None


def hub_pin(repo_id: str | None, token: str | None = None) -> HubPin:
    """The commit to pin and the bytes to download for a Hub model, refusing one that cannot be used.

    Without a token only public, ungated models pass. With the owner's, a private or gated one their
    account can read passes too."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

    if not repo_id:
        raise CustomModelFileError("A Hub model needs its repository id")
    try:
        info = HfApi().model_info(repo_id, files_metadata=True, token=_hub_token(token))
    except RevisionNotFoundError:
        raise CustomModelFileError(f"'{repo_id}' has no default revision") from None
    except HfHubHTTPError as e:
        if (refusal := _access_error(repo_id, e, token)) is not None:
            raise refusal from None
        raise
    if info.private and not token:
        raise CustomModelFileError(
            f"'{repo_id}' is private. Connect your Hugging Face account in Settings to import it"
        )
    if not info.sha:
        raise CustomModelFileError(f"Could not determine the current revision of '{repo_id}'")

    siblings = info.siblings or []
    wanted = [
        s
        for s in siblings
        if not any(s.rfilename.lower().endswith(ext) for ext in FORBIDDEN_EXTENSIONS)
        and s.rfilename.lower().endswith((".safetensors", ".json", ".txt", ".model", ".tiktoken", ".vocab", ".md"))
    ]
    return HubPin(
        revision=info.sha,
        size_bytes=sum(s.size or 0 for s in wanted),
        has_safetensors=any(s.rfilename.lower().endswith(".safetensors") for s in wanted),
        has_config=any(s.rfilename == "config.json" for s in siblings),
        private=bool(info.private or info.gated),
    )


def fetch_hub_config(repo_id: str, revision: str, token: str | None = None) -> dict | None:
    """The repo's top-level config.json at `revision`, or None when it has none or is not a JSON object."""
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    try:
        path = Path(hf_hub_download(repo_id, "config.json", revision=revision, token=_hub_token(token)))
    except EntryNotFoundError:
        return None
    except HfHubHTTPError as e:
        if (refusal := _access_error(repo_id, e, token)) is not None:
            raise refusal from None
        raise
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise CustomModelFileError("config.json is unreasonably large")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CustomModelFileError("config.json could not be read as JSON") from None
    return config if isinstance(config, dict) else None


def _fetch_hub(row: CustomModel, dest: Path, token: str | None = None) -> None:
    if not row.source_ref or not row.revision:
        raise CustomModelFileError("This Hub model has not been validated yet, so it has no pinned revision")
    from huggingface_hub import snapshot_download  # lazy: only Hub models need it
    from huggingface_hub.errors import HfHubHTTPError

    try:
        snapshot_download(
            repo_id=row.source_ref,
            revision=row.revision,
            local_dir=str(dest),
            allow_patterns=list(_HUB_ALLOW),
            ignore_patterns=[f"*{ext}" for ext in sorted(FORBIDDEN_EXTENSIONS)],
            token=_hub_token(token),
        )
    except HfHubHTTPError as e:
        if (refusal := _access_error(row.source_ref, e, token)) is not None:
            raise refusal from None
        raise


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
            convert_pth_weights(dest)
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


def convert_pth_weights(root: Path) -> None:
    """Turn a `.pth` in an unpacked bundle into `model.safetensors` beside it, so the rest of the pipeline
    only ever sees safetensors. A .pth is a pickle, so it is read with `weights_only=True` (tensors and plain
    containers only, no code) and anything else in it is refused. Only a plain state dict (optionally wrapped
    in `{"state_dict": ...}`) is accepted; the architecture comes from the bundle's config.json.
    """
    found = sorted(root.rglob("*.pth"))
    if not found:
        return
    if len(found) > 1:
        raise CustomModelFileError("The zip has more than one .pth file; include exactly one set of weights")
    source = found[0]
    target = source.with_name("model.safetensors")
    if any(root.rglob("*.safetensors")):
        raise CustomModelFileError("The zip has both .pth and .safetensors weights; include only one of them")

    import torch  # lazy: only a .pth upload needs it
    from safetensors.torch import save_file

    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  anything the safe unpickler refuses is not a plain state dict
        raise CustomModelFileError(
            f"'{source.name}' is not a plain PyTorch state dict (it holds more than tensors), "
            "so it cannot be loaded safely. Convert the weights to .safetensors"
        ) from None
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CustomModelFileError(f"'{source.name}' is not a state dict of tensors")
    # safetensors refuses tensors that share memory and non-contiguous ones; a copy of each is always valid.
    tensors = {str(k): v.detach().contiguous().clone() for k, v in state.items()}
    save_file(tensors, str(target), metadata={"format": "pt"})
    source.unlink()


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
