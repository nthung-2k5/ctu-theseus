"""S3 access and the canonical key layout, merged from server/lib/storage.ts and ai_service/services/storage.py.

Every function here is synchronous (boto3). Call them from async code via an executor, never
directly on the event loop.

theseus-datasets/
  pool/{projectId}/{hash[0:2]}/{hash}{ext}   content-addressed, deduplicated upload pool
  snapshots/{versionId}/dataset.parquet      immutable, cut from the pool
  snapshots/{versionId}/manifest.json
  snapshots/{versionId}/augmented/{hh}/{hash}{ext}   augmented copies built with that snapshot (not pooled)
theseus-training/
  {runId}/config.yaml                        compiled Ludwig config
  {runId}/results/                           Ludwig output tree (incl. training_set_metadata.json)
  {runId}/logs/train.log
  {runId}/evaluation/report.json             bounded evaluation report
  {runId}/evaluation/predictions.parquet     full per-row predictions
theseus-models/
  {runId}/model.{onnx|pt2}                   converted export artifacts (shared by formats)
  {runId}/bundles/{exportId}.zip             assembled devkit/app bundles
  {runId}/expected.json                      golden sample for verify scripts
  {runId}/predictions/{inferenceId}.csv      batch inference results
theseus-uploads/
  inference/{inferenceId}/input{ext}         inference inputs that outlive the request
"""

import hashlib
import json
import logging
import os
import re
import shutil
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from theseus import constants as C
from theseus.settings import get_settings

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

BUCKETS = (C.BUCKET_DATASETS, C.BUCKET_TRAINING, C.BUCKET_MODELS, C.BUCKET_UPLOADS)


@lru_cache
def s3(endpoint_url: str | None = None) -> "S3Client":
    """Created lazily so importing the app never needs S3 to be reachable."""
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name="us-east-1",
        # Explicit SigV4. Against a custom endpoint boto3 signs presigned URLs the legacy SigV2 way
        # (AWSAccessKeyId/Signature/Expires), which RustFS rejects with 403 SignatureDoesNotMatch.
        config=Config(signature_version="s3v4"),
    )


def ensure_buckets() -> None:
    """Create required buckets if they do not exist (idempotent)."""
    for bucket in BUCKETS:
        try:
            s3().head_bucket(Bucket=bucket)
        except ClientError:
            try:
                s3().create_bucket(Bucket=bucket)
                logger.info("Bucket %s created.", bucket)
            except Exception as e:
                logger.warning("Could not create bucket %s: %s", bucket, e)


# -- Key builders ----------------------------------------------------------------------------


def pool_key(project_id: str, content_hash: str, ext: str) -> str:
    return f"pool/{project_id}/{content_hash[:2]}/{content_hash}{ext}"


def snapshot_parquet_key(version_id: str) -> str:
    return f"snapshots/{version_id}/{C.DATASET_VERSION_FILENAME}"


def snapshot_manifest_key(version_id: str) -> str:
    return f"snapshots/{version_id}/manifest.json"


def augmented_prefix(version_id: str) -> str:
    """Everything a snapshot's augmentation wrote. Version-scoped on purpose, NOT in the content-addressed
    pool: an augmented copy belongs to exactly one snapshot, so deleting the snapshot deletes the prefix
    and no copy can be shared with (or deduplicated onto) a real pool item."""
    return f"snapshots/{version_id}/augmented/"


def augmented_key(version_id: str, content_hash: str, ext: str) -> str:
    return f"{augmented_prefix(version_id)}{content_hash[:2]}/{content_hash}{ext}"


def training_config_key(run_id: str) -> str:
    return f"{run_id}/{C.TRAINING_CONFIG_FILENAME}"


def training_results_prefix(run_id: str) -> str:
    return f"{run_id}/results/"


def training_logs_key(run_id: str) -> str:
    return f"{run_id}/logs/train.log"


def evaluation_prefix(run_id: str) -> str:
    return f"{run_id}/evaluation/"


def evaluation_report_key(run_id: str) -> str:
    return f"{evaluation_prefix(run_id)}report.json"


def evaluation_predictions_key(run_id: str) -> str:
    return f"{evaluation_prefix(run_id)}predictions.parquet"


def export_key(run_id: str, artifact_filename: str) -> str:
    """The converted model artifact, e.g. `{run}/model.onnx`. Shared by every format built from it."""
    return f"{run_id}/{artifact_filename}"


def bundle_key(run_id: str, export_id: str) -> str:
    return f"{run_id}/bundles/{export_id}.zip"


def expected_sample_key(run_id: str) -> str:
    return f"{run_id}/expected.json"


def batch_inference_result_key(run_id: str, inference_id: str) -> str:
    return f"{run_id}/predictions/{inference_id}.csv"


def inference_upload_key(inference_id: str, ext: str) -> str:
    return f"inference/{inference_id}/input{ext}"


def s3fs_path(bucket: str, key: str) -> str:
    """A path Ludwig/pandas can open through s3fs."""
    return f"s3://{bucket}/{key}"


# -- Object operations -----------------------------------------------------------------------


def get_download_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    """A time-limited URL for the BROWSER (dataset thumbnails, export/log/batch downloads).

    Signed against `s3_public_endpoint`, not the endpoint this process uses for its own S3 calls: the
    signature covers the Host header and the URL embeds the host, so a URL signed for a container-network
    hostname is useless to a browser on the host. Every other operation here keeps using `s3_endpoint`.
    """
    s = get_settings()
    client = s3(s.s3_public_endpoint)  # None falls back to s3_endpoint
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str | None = None) -> None:
    extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}
    s3().put_object(Bucket=bucket, Key=key, Body=data, **extra)


def download_bytes(bucket: str, key: str) -> bytes:
    return s3().get_object(Bucket=bucket, Key=key)["Body"].read()


def file_exists(bucket: str, key: str) -> bool:
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def delete_file(bucket: str, key: str) -> None:
    s3().delete_object(Bucket=bucket, Key=key)


def delete_files(bucket: str, keys: list[str]) -> None:
    """Delete many objects in batches of 1000 (the S3 DeleteObjects limit)."""
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i : i + 1000]]
        s3().delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})


def list_keys(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    for page in s3().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def delete_prefix(bucket: str, prefix: str) -> int:
    """Delete everything under a prefix in batches of 1000 (the S3 DeleteObjects limit)."""
    keys = list_keys(bucket, prefix)
    delete_files(bucket, keys)
    return len(keys)


class PoolUpload(NamedTuple):
    key: str
    hash: str
    byte_size: int
    is_duplicate: bool


def upload_to_pool(project_id: str, data: bytes, ext: str, content_type: str | None = None) -> PoolUpload:
    """Upload to the project's content-addressed pool, skipping the write if that hash already exists.

    Every dataset upload path must go through this, or per-project dedup silently breaks.
    """
    digest = hashlib.sha256(data).hexdigest()
    key = pool_key(project_id, digest, ext)
    existing = file_exists(C.BUCKET_DATASETS, key)
    if not existing:
        upload_bytes(C.BUCKET_DATASETS, key, data, content_type)
    return PoolUpload(key, digest, len(data), existing)


# -- Local files and Ludwig model directories ------------------------------------------------


def download_file(bucket: str, key: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3().download_file(bucket, key, local_path)
    return local_path


def upload_file(bucket: str, key: str, local_path: str) -> None:
    s3().upload_file(local_path, bucket, key)


def upload_json(bucket: str, key: str, data: Any) -> None:
    """Serialize and upload as a JSON object (no temp file)."""
    upload_bytes(bucket, key, json.dumps(data, indent=2, default=str).encode("utf-8"), "application/json")


def download_prefix(bucket: str, prefix: str, local_dir: str) -> int:
    """Download every object under a prefix into local_dir, keeping relative paths."""
    count = 0
    for key in list_keys(bucket, prefix):
        rel = key[len(prefix) :].lstrip("/")
        if not rel:
            continue
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3().download_file(bucket, key, local_path)
        count += 1
    logger.info("Downloaded %d files from s3://%s/%s", count, bucket, prefix)
    return count


_DOWNLOAD_COMPLETE_MARKER = ".download_complete"


def download_model(run_id: str) -> str:
    """Download a run Ludwig output directory to a local cache and return its path.

    Cache validity is a marker file written only after the download finishes, not merely a
    non-empty directory, so a download that died partway is never mistaken for a cache hit.
    """
    local_dir = str(get_settings().temp_dir / "models" / run_id)
    marker = os.path.join(local_dir, _DOWNLOAD_COMPLETE_MARKER)
    if not os.path.isfile(marker):
        shutil.rmtree(local_dir, ignore_errors=True)
        download_prefix(C.BUCKET_TRAINING, training_results_prefix(run_id), local_dir)
        with open(marker, "w"):
            pass
    return local_dir


_RUN_SUFFIX_RE = re.compile(r"_run_(\d+)$")


def _run_ordinal(dirpath: str) -> int:
    """Ludwig results_run_N ordinal for any ancestor of dirpath, else 0 (plain results is the first run)."""
    best = 0
    for part in os.path.normpath(dirpath).split(os.sep):
        match = _RUN_SUFFIX_RE.search(part)
        if match:
            best = max(best, int(match.group(1)))
    return best


def find_model_dir(root: str) -> str:
    """Find the loadable model under a downloaded results tree.

    Ludwig nests it under a run directory it names itself (results_run_N/model/), so walk for
    the marker file it always writes next to a loadable model. When several exist take the
    highest-numbered run: Ludwig increments rather than overwrites, and os.walk order would
    otherwise decide which checkpoint gets served.
    """
    candidates = [d for d, _, files in os.walk(root) if "model_hyperparameters.json" in files]
    return max(candidates, key=_run_ordinal) if candidates else root


def cleanup_temp(subdir: str, job_id: str) -> None:
    """Remove a local temp working directory for a finished job."""
    shutil.rmtree(get_settings().temp_dir / subdir / job_id, ignore_errors=True)
