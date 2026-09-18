import json
import logging
import os
import re
import shutil
from typing import Any
from uuid import UUID

import boto3
from config import S3_ACCESS_KEY, S3_ENDPOINT, S3_SECRET_KEY, TEMP_DIR
from constants import BUCKET_DATASETS, BUCKET_MODELS, BUCKET_TRAINING
from types_boto3_s3.type_defs import ObjectIdentifierTypeDef

logger = logging.getLogger(__name__)

# Exported model artifacts (onnx/torchscript) are stored alongside trained
# models. Revisited in the S3 layout rework (models bucket becomes export-only).
BUCKET_EXPORTS = BUCKET_MODELS

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name="us-east-1",
)


def ensure_buckets():
    """Create required buckets if they don't exist (idempotent)."""
    for bucket in [BUCKET_DATASETS, BUCKET_TRAINING, BUCKET_MODELS]:
        try:
            s3.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket '{bucket}' exists.")
        except s3.exceptions.ClientError:
            try:
                s3.create_bucket(Bucket=bucket)
                logger.info(f"Bucket '{bucket}' created.")
            except Exception as e:
                logger.warning(f"Could not create bucket '{bucket}': {e}")


def download_prefix(bucket: str, prefix: str, local_dir: str) -> int:
    """
    Download all objects under a prefix to a local directory.
    Returns the number of files downloaded.
    """
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key") or ""
            # Compute relative path from prefix
            rel_path = key[len(prefix) :].lstrip("/")
            if not rel_path:
                continue

            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            count += 1

    logger.info(f"Downloaded {count} files from s3://{bucket}/{prefix} → {local_dir}")
    return count


def download_file(bucket: str, key: str, local_path: str) -> str:
    """Download a single file from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    logger.debug(f"Downloaded s3://{bucket}/{key} → {local_path}")
    return local_path


def upload_directory(bucket: str, prefix: str, local_dir: str) -> int:
    """
    Upload all files in a local directory to S3 under a prefix.
    Returns the number of files uploaded.
    """
    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = f"{prefix.rstrip('/')}/{rel_path}"
            s3.upload_file(local_path, bucket, key)
            count += 1

    logger.info(f"Uploaded {count} files from {local_dir} → s3://{bucket}/{prefix}")
    return count


def upload_file(bucket: str, key: str, local_path: str):
    """Upload a single file to S3."""
    s3.upload_file(local_path, bucket, key)
    logger.debug(f"Uploaded {local_path} → s3://{bucket}/{key}")


def upload_json(bucket: str, key: str, data: Any):
    """Serialize `data` and upload it directly as a JSON object (no temp file)."""
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.debug(f"Uploaded JSON → s3://{bucket}/{key}")


def delete_prefix(bucket: str, prefix: str):
    """Delete all objects under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects: list[ObjectIdentifierTypeDef] = [
            {"Key": obj.get("Key") or ""} for obj in page.get("Contents", [])
        ]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})


def delete_file(bucket: str, key: str):
    """Delete a single file from S3."""
    s3.delete_object(Bucket=bucket, Key=key)


def file_exists(bucket: str, key: str) -> bool:
    """Check if a file exists in S3."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def s3fs_readable_path(bucket: str, key: str) -> str:
    """Return a path that can be opened with s3fs."""
    return f"s3://{bucket}/{key}"


# ──────────────────────────────────────────────────────────────────
# High-level helpers for common operations
# ──────────────────────────────────────────────────────────────────


def training_logs_key(run_id: UUID) -> str:
    return f"{run_id}/logs/train.log"


def evaluation_report_key(run_id: UUID | str) -> str:
    return f"{run_id}/evaluation/report.json"


def evaluation_predictions_key(run_id: UUID | str) -> str:
    return f"{run_id}/evaluation/predictions.parquet"


def batch_inference_result_key(run_id: UUID | str, inference_id: UUID | str) -> str:
    return f"{run_id}/predictions/{inference_id}.csv"


_DOWNLOAD_COMPLETE_MARKER = ".download_complete"


def download_model(run_id: str) -> str:
    """
    Download a training run's Ludwig output directory (checkpoint,
    metadata, hyperparameters — everything LudwigModel.load() needs) to a
    local cache and return its path. Skips re-downloading if already cached
    locally.

    Cache validity is a marker file written only after `download_prefix`
    returns successfully — not just "the directory is non-empty" (the
    previous check), which would treat a directory left behind by a
    download that failed partway through as a valid, permanent cache hit.
    On a miss, any partial contents from a prior failed attempt are cleared
    before retrying.
    """
    local_dir = str(TEMP_DIR / "models" / run_id)
    marker = os.path.join(local_dir, _DOWNLOAD_COMPLETE_MARKER)
    if not os.path.isfile(marker):
        shutil.rmtree(local_dir, ignore_errors=True)
        download_prefix(BUCKET_TRAINING, f"{run_id}/results/", local_dir)
        with open(marker, "w") as f:
            f.write("")
    return local_dir


_RUN_SUFFIX_RE = re.compile(r"_run_(\d+)$")


def _run_ordinal(dirpath: str) -> int:
    """Ludwig's `results_run_N` ordinal for any ancestor of `dirpath`, else 0.

    `results` (no suffix) is Ludwig's first run and sorts below `results_run_1`.
    """
    best = 0
    for part in os.path.normpath(dirpath).split(os.sep):
        match = _RUN_SUFFIX_RE.search(part)
        if match:
            best = max(best, int(match.group(1)))
    return best


def find_model_dir(root: str) -> str:
    """
    Ludwig nests the actual saved model under an experiment-run
    subdirectory it names itself (`results_run_N/model/`), so the layout
    under `download_model()`'s local cache isn't fixed. Walk for the
    marker file Ludwig always writes next to a loadable model.

    When several are present, take the highest-numbered run. Ludwig
    auto-increments `results_run_N` rather than overwriting, so a redelivered
    or re-dispatched training run leaves earlier attempts in place — and
    `os.walk` order would otherwise decide which checkpoint export and
    inference load, silently serving an abandoned partial model.
    """
    candidates = [
        dirpath
        for dirpath, _, filenames in os.walk(root)
        if "model_hyperparameters.json" in filenames
    ]
    if not candidates:
        return root
    return max(candidates, key=_run_ordinal)


def cleanup_temp(subdir: str, job_id: str) -> None:
    """Remove a local temp working directory for a completed job."""
    shutil.rmtree(TEMP_DIR / subdir / job_id, ignore_errors=True)
