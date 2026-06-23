"""
S3-compatible storage client for the AI worker.

Uses boto3 to communicate with RustFS (or any S3-compatible server).
Handles downloading datasets before training and uploading model artifacts after.
"""

import logging
import os
from pathlib import Path
from uuid import UUID

import boto3

from ai_service.constants import (
    BUCKET_DATASETS,
    BUCKET_MODELS,
    BUCKET_TRAINING,
    DATASET_VERSION_FILENAME,
    TRAINING_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "theseus")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "theseus-secret")

# Local temp directories for in-flight data
TEMP_DIR = Path(os.environ.get("TEMP_DIR", "/tmp/theseus"))

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
            key = obj["Key"]
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


def delete_prefix(bucket: str, prefix: str):
    """Delete all objects under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
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


def get_training_run_config_path(run_id: UUID) -> str | None:
    """Get the training run config path in the training bucket."""
    path = f"{run_id}/{TRAINING_CONFIG_FILENAME}"

    if not file_exists(BUCKET_TRAINING, path):
        return None

    return s3fs_readable_path(BUCKET_TRAINING, path)


def get_training_output_path(run_id: UUID) -> str:
    """Return the training run output path in the models bucket."""
    return s3fs_readable_path(BUCKET_MODELS, f"{run_id}/")


def get_training_dataset_path(version_id: UUID) -> str | None:
    """Return the dataset path in the datasets bucket."""
    path = f"snapshots/{version_id}/{DATASET_VERSION_FILENAME}"

    if not file_exists(BUCKET_DATASETS, path):
        return None

    return s3fs_readable_path(BUCKET_DATASETS, path)
