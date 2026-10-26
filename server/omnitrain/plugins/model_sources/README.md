# Model sources

A model source says where a bring-your-own model's files come from. Two ship today: a Hugging Face Hub repository
fetched at a pinned commit (`hub.py`) and a zip the user uploads (`upload.py`). `custom_models.source_kind` is free
text holding the source id.

## How a source is found

`Registry` imports every module directly under `omnitrain.plugins.model_sources` whose name does not start with
`_`. A `ModelSource` subclass registers itself when created, but only if it sets its **own** `id` (it must also set
a `label`, or class creation raises `TypeError`). A duplicate `id` raises at import time.

## Adding a source (ModelScope, a plain URL, another bucket)

1. **Create a module** `model_sources/<name>.py`.
2. **Subclass `ModelSource`** and set `id` and `label`, then the flags that apply:
   - `needs_ref = True` if the user has to name the model (a repository id). Override `normalize_ref(ref)` to clean
     and check it; raise `SourceRefError` with a message that is safe to show.
   - `needs_upload = True` if the model is created waiting for the browser to upload a bundle.
   - `shared_failure_hint`, a sentence appended when validation fails for a model shared with everyone (which never
     carries one person's credential).
3. **Implement `validate(row, token, limit)`**: first contact. Pin the revision, bring the files to disk, hash them,
   and return `(Validated(...), directory)`. The directory is then judged by the shared safety checks.
4. **Implement `fetch(row, dest, token)`**: put the model's files in the empty directory `dest`. Raise
   `CustomModelFileError` when you cannot.
5. **Optional**:
   - `token_for(owner_user_id)` returns the owner's saved credential for this source, or `None`.
   - `worker_fetchable(model)` returns a SQL condition saying a remote worker may fetch the model with no secret.
     The default is `sa.false()`.
6. **Allow the source for a kind.** A trainer backend lists the sources a kind may use in
   `CustomModelKind.source_kinds`; add your id there for the kinds that should accept it.
7. **The web import form** only knows how to collect a Hub id or a file. A source with another input needs a form
   there to be offered to people; the backend side needs nothing more.
8. **Test** it like `server/tests/test_model_sources.py`.

## Minimal example

```python
# model_sources/https_zip.py
from pathlib import Path

from omnitrain.plugins.model_sources.base import ModelSource, SourceRefError, Validated


class HttpsZip(ModelSource):
    id = "https_zip"
    label = "Zip at an https URL"
    needs_ref = True

    @classmethod
    def normalize_ref(cls, ref):
        ref = (ref or "").strip()
        if not ref.startswith("https://"):
            raise SourceRefError("Give an https:// URL")
        return ref

    @classmethod
    def fetch(cls, row, dest: Path, token): ...  # download row.source_ref and unpack it into dest

    @classmethod
    def validate(cls, row, token, limit):
        ...  # download into a temp dir, hash it, then:
        # row.sha256 = sha
        # return Validated(sha256=sha, size_bytes=size), directory
```

## What a source never decides

These are the same for every source and live in `services/custom_models.py`. Do not reimplement them:

- the deterministic local path and the completion marker;
- `check_tree`: no pickles, no code, no symlinks, no zip-slip;
- the sha256 re-check and the size limit.

A source only puts files at the path it is given. Files reach disk only through `custom_models.ensure_local` and
`import_upload`.

## Rules that are easy to break

- **Credentials**: a source uses only the model **owner's** saved token (`token_for`), never anyone else's and
  never a server-wide one. For the Hub, every call goes through `custom_models._hub_token`, which turns "no token"
  into `False`; passing `None` would let `huggingface_hub` fall back to `HF_TOKEN` or a cached login on the server.
  Never log or return a token.
- **Workers**: a remote worker (`omnitrain/worker/`) runs the same `fetch` with no database and no secret, and
  must run the same package version. Keep `worker_fetchable` at its default (`False`) unless `fetch` needs nothing
  but S3 and the public internet.
- **A model is immutable** once created (backend, kind, source, revision, checksum), and a model a run used is
  archived, never deleted (`training_runs.custom_model_id` is `ON DELETE RESTRICT`). Pin a revision or hash so the
  files a run trained on can be fetched again.
- **Never trust a `custom:{uuid}` id from a request.** Visibility and use are decided in
  `services/model_catalog.resolve`, and ownership in `services/custom_model_ops.py`.
- **Layering**: a source may use `services.custom_models`; it must not import `api/`.
