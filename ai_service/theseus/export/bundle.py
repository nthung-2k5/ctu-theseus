"""Assemble an export bundle zip and upload it (ported from server/lib/export/bundle.ts).

assemble_files() is pure (bytes in, file list out) so the per-tier, per-language layout can be
tested exhaustively; build_bundle() is the thin async wrapper that loads the inputs, zips in the
export executor, uploads, and owns the final `ready` / `failed` transition. It never raises.
"""

import asyncio
import hashlib
import io
import json
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from theseus import constants as C
from theseus.db.base import get_sessionmaker
from theseus.db.models import ModelExport, Project, TrainingRun
from theseus.export.metadata import extract_preprocessing
from theseus.export.readme import ReadmeVars, render_readme
from theseus.jobs.executors import export_executor, run_in_executor
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"

DEVKIT_LANGS = ("python", "typescript", "csharp", "java")
APP_LANGS = ("pwa", "flutter")

# A fixed timestamp keeps a bundle reproducible: the same inputs always produce the same zip bytes.
ZIP_EPOCH = (2020, 1, 1, 0, 0, 0)


@dataclass
class BundleFile:
    path: str
    data: bytes
    # The model artifact is already compact binary: store it (level 0) instead of deflating.
    compress: bool = True


@dataclass
class GoldenSample:
    expected_json: bytes
    sample_filename: str
    sample_bytes: bytes


@lru_cache
def template(relative: str) -> str:
    return (TEMPLATES / relative).read_text(encoding="utf-8")


def render(tpl: str, variables: dict[str, str]) -> str:
    """{{VAR}} substitution for README and manifest-ish files: deliberately a literal replace, not a template engine."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: variables.get(m.group(1), ""), tpl)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def dart_package_name(name: str) -> str:
    """A valid pubspec `name:`: lowercase_with_underscores, starting with a letter."""
    snake = re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9]+", "_", name.lower()))
    if not snake:
        return "theseus_app"
    return snake if re.match(r"[a-z]", snake) else f"app_{snake}"


def resolve_sample_file(input_value: Any, download) -> tuple[str, bytes] | None:
    """Turn expected.json's raw inputValue into an actual file to embed under sample/.

    An s3:// URI (file-backed modalities) is downloaded; a string becomes input.txt; anything else
    becomes input.json. download(bucket, key) may raise, which yields None (no sample, so no verify).
    """
    if isinstance(input_value, str) and input_value.startswith("s3://"):
        rest = input_value[len("s3://") :]
        if "/" not in rest:
            return None
        bucket, key = rest.split("/", 1)
        ext = key[key.rfind(".") :] if "." in key else ""
        try:
            return f"input{ext}", download(bucket, key)
        except Exception:
            return None
    if isinstance(input_value, str):
        return "input.txt", input_value.encode()
    if input_value is not None:
        return "input.json", dumps(input_value).encode()
    return None


def to_bundle_local(expected: dict[str, Any], sample_filename: str) -> bytes:
    """expected.json rewritten to reference the sample by its bundle-local path."""
    return dumps(
        {
            "schemaVersion": 1,
            "sampleFile": f"sample/{sample_filename}",
            "outputColumn": expected.get("outputColumn"),
            "outputType": expected.get("outputType"),
            "predictions": expected.get("predictions"),
        }
    ).encode()


def assemble_files(
    *,
    run_name: str,
    task_label: str,
    tier: str,
    fmt: str,
    lang: str | None,
    model_bytes: bytes,
    preprocessing: dict[str, Any],
    golden: GoldenSample | None,
) -> list[BundleFile]:
    files: list[BundleFile] = []
    labels = next((o["classes"] for o in preprocessing["outputs"] if o.get("classes")), None)
    pre_json = dumps(preprocessing).encode()

    def add(path: str, data: str | bytes, compress: bool = True) -> None:
        files.append(BundleFile(path, data.encode() if isinstance(data, str) else data, compress))

    def place_model(prefix: str) -> None:
        add(f"{prefix}model.{fmt}", model_bytes, compress=False)
        add(f"{prefix}preprocessing.json", pre_json)
        if labels:
            add(f"{prefix}labels.txt", "\n".join(labels))

    def place_golden(prefix: str) -> None:
        if golden is None:
            return
        add(f"{prefix}expected.json", golden.expected_json)
        add(f"{prefix}sample/{golden.sample_filename}", golden.sample_bytes)

    if tier == "model":
        place_model("")
        add("README.md", render_readme(ReadmeVars(run_name, task_label, fmt, tier, False), None))
        return files

    # devkit / app: the model tier files plus a generated client.
    allowed = {"devkit": DEVKIT_LANGS, "app": APP_LANGS}.get(tier, ())
    if lang not in allowed:
        raise ValueError(f"tier '{tier}' requires a lang in {sorted(allowed)}, got {lang!r}")
    variables = {"RUN_NAME": run_name, "TASK": task_label}
    readme = ReadmeVars(run_name, task_label, fmt, tier, golden is not None)

    if lang == "python":
        place_model("")
        place_golden("")
        add("theseus_client.py", template("python/theseus_client.py"))
        add("example.py", template("python/example.py"))
        if golden:
            add("verify.py", template("python/verify.py"))
    elif lang == "typescript":
        place_model("")
        place_golden("")
        add("client.ts", template("typescript/client.ts.tmpl"))
        add("example.ts", template("typescript/example.ts.tmpl"))
        if golden:
            add("verify.ts", template("typescript/verify.ts.tmpl"))
    elif lang == "csharp":
        place_model("")
        place_golden("")
        add("TheseusClient.cs", template("csharp/TheseusClient.cs"))
        add("Program.cs", template("csharp/Program.cs"))
    elif lang == "java":
        place_model("")
        place_golden("")
        add("TheseusClient.java", template("java/TheseusClient.java"))
        add("Main.java", template("java/Main.java"))
    elif lang == "pwa":
        # Fetched relative to index.html at runtime, so root placement is correct.
        place_model("")
        place_golden("")
        add("index.html", render(template("pwa/index.html.tmpl"), variables))
        add("app.js", template("pwa/app.js"))
        add("sw.js", template("pwa/sw.js"))
        add("manifest.webmanifest", render(template("pwa/manifest.webmanifest.tmpl"), variables))
        add("icon.svg", template("pwa/icon.svg"))
        add("style.css", template("pwa/style.css"))
    elif lang == "flutter":
        # Flutter only reads files declared as pubspec assets, so the assets/ prefix is required.
        place_model("assets/")
        place_golden("assets/")
        add(
            "pubspec.yaml",
            render(template("flutter/pubspec.yaml.tmpl"), {**variables, "PACKAGE_NAME": dart_package_name(run_name)}),
        )
        add("lib/main.dart", render(template("flutter/lib/main.dart.tmpl"), variables))
        add("lib/theseus_client.dart", template("flutter/lib/theseus_client.dart"))
    else:  # unreachable while DEVKIT_LANGS/APP_LANGS match the branches above
        raise AssertionError(lang)

    add("README.md", render_readme(readme, lang))
    return files


def make_zip(files: list[BundleFile]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in files:
            info = zipfile.ZipInfo(f.path, date_time=ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            zf.writestr(
                info,
                f.data,
                compress_type=zipfile.ZIP_DEFLATED if f.compress else zipfile.ZIP_STORED,
                compresslevel=6,
            )
    return buf.getvalue()


# -- Orchestration ---------------------------------------------------------------------------


async def _load_golden(run_id: uuid.UUID) -> GoldenSample | None:
    """The worker-written expected.json (if any) and its resolved sample file, with bundle-local paths."""
    loop = asyncio.get_running_loop()
    key = storage.expected_sample_key(str(run_id))
    if not await loop.run_in_executor(None, storage.file_exists, C.BUCKET_MODELS, key):
        return None
    expected = json.loads(await loop.run_in_executor(None, storage.download_bytes, C.BUCKET_MODELS, key))
    sample = await loop.run_in_executor(None, resolve_sample_file, expected.get("inputValue"), storage.download_bytes)
    if sample is None:
        return None
    filename, data = sample
    return GoldenSample(to_bundle_local(expected, filename), filename, data)


async def _assemble(export_id: uuid.UUID) -> tuple[uuid.UUID, list[BundleFile]]:
    async with get_sessionmaker()() as session:
        row = await session.get(ModelExport, export_id)
        if row is None:
            raise ValueError(f"Export {export_id} not found")
        run = await session.get(TrainingRun, row.run_id)
        project = await session.get(Project, run.project_id) if run else None
        if run is None or project is None:
            raise ValueError(f"Run {row.run_id} or its project not found")
        preprocessing = await extract_preprocessing(session, run.id)
        tier, fmt, lang = row.tier, row.format, row.lang
        run_id, run_name = run.id, run.name
        task_label = get_task_descriptor(project.task).label

    model_bytes = await run_in_executor(
        None, storage.download_bytes, C.BUCKET_MODELS, storage.export_key(str(run_id), fmt)
    )
    golden = await _load_golden(run_id) if tier != "model" else None
    files = assemble_files(
        run_name=run_name, task_label=task_label, tier=tier, fmt=fmt, lang=lang,
        model_bytes=model_bytes, preprocessing=preprocessing, golden=golden,
    )  # fmt: skip
    return run_id, files


async def build_bundle(export_id: uuid.UUID) -> None:
    """Build one export zip, upload it, and flip assembling -> ready | failed. Never raises.

    Both transitions are guarded on status = 'assembling': if startup recovery or a lease reaper
    already moved the row on, this result is stale and must not overwrite it.
    """
    try:
        run_id, files = await _assemble(export_id)
        zipped = await run_in_executor(export_executor, make_zip, files)
        key = storage.bundle_key(str(run_id), str(export_id))
        await run_in_executor(None, storage.upload_bytes, C.BUCKET_MODELS, key, zipped, "application/zip")
        checksum = hashlib.sha256(zipped).hexdigest()
        values: dict[str, Any] = {
            "status": "ready", "bundle_key": key, "byte_size": len(zipped), "checksum": checksum,
            "ready_at": sa.func.now(), "claimed_by": None, "lease_expires_at": None,
        }  # fmt: skip
    except Exception as e:
        logger.exception("Export assembly failed for %s", export_id)
        values = {
            "status": "failed", "failed_message": str(e)[:500], "claimed_by": None, "lease_expires_at": None,
        }  # fmt: skip
    try:
        async with get_sessionmaker()() as session:
            await session.execute(
                sa.update(ModelExport)
                .where(ModelExport.id == export_id, ModelExport.status == "assembling")
                .values(**values)
            )
            await session.commit()
    except Exception:
        logger.exception("Could not record the assembly outcome for export %s", export_id)
