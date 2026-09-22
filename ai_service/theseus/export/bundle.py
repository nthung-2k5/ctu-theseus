"""Assemble an export bundle zip and upload it (ported from server/lib/export/bundle.ts).

assemble_files() is pure (bytes in, file list out) and delegates the layout to the export format
class (theseus/export/formats/), so each format's layout can be tested on its own. build_bundle() is
the thin async wrapper that loads the inputs, zips in the export executor, uploads, and owns the
final `ready` / `failed` transition. It never raises.
"""

import asyncio
import hashlib
import io
import json
import logging
import uuid
import zipfile
from typing import Any

import sqlalchemy as sa

from theseus import constants as C
from theseus.backends.base import ARTIFACT_FILENAMES
from theseus.db.base import get_sessionmaker
from theseus.db.models import ModelExport, Project, TrainingRun
from theseus.export.common import TEMPLATES, BundleFile, GoldenSample, dart_package_name, dumps, render, template
from theseus.export.formats.base import BundleContext
from theseus.export.metadata import extract_preprocessing
from theseus.export.registry import get_export_format
from theseus.jobs.executors import export_executor, run_in_executor
from theseus.services import storage
from theseus.services.task_registry import get_task_descriptor

logger = logging.getLogger(__name__)

# A fixed timestamp keeps a bundle reproducible: the same inputs always produce the same zip bytes.
ZIP_EPOCH = (2020, 1, 1, 0, 0, 0)

__all__ = ["TEMPLATES", "BundleFile", "GoldenSample", "dart_package_name", "dumps", "render", "template"]


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
    format_id: str,
    model_bytes: bytes,
    preprocessing: dict[str, Any],
    golden: GoldenSample | None,
) -> list[BundleFile]:
    fmt = get_export_format(format_id)
    ctx = BundleContext(
        run_name=run_name,
        task_label=task_label,
        format_label=fmt.label,
        artifact_filename=ARTIFACT_FILENAMES[fmt.artifact],
        model_bytes=model_bytes,
        preprocessing=preprocessing,
        golden=golden if fmt.needs_golden else None,
    )
    fmt.assemble(ctx)
    return ctx.files


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
        format_id = row.format
        run_id, run_name = run.id, run.name
        task_label = get_task_descriptor(project.task).label

    fmt = get_export_format(format_id)
    model_bytes = await run_in_executor(
        None, storage.download_bytes, C.BUCKET_MODELS, storage.export_key(str(run_id), ARTIFACT_FILENAMES[fmt.artifact])
    )
    golden = await _load_golden(run_id) if fmt.needs_golden else None
    files = assemble_files(
        run_name=run_name, task_label=task_label, format_id=format_id,
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
