"""Turn a snapshot's dataset.parquet into local clip files and a labelled sample list per split.

Like Ultralytics, OpenCV reads local files only, so every usable clip is downloaded into the run's workdir first.
Imports pandas only: no torch, no cv2.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from omnitrain.services import storage

logger = logging.getLogger(__name__)

# Snapshot split name -> the key used here.
SPLITS = {"train": "train", "validation": "val", "test": "test"}
DOWNLOAD_WORKERS = 4


@dataclass(frozen=True)
class Sample:
    path: str
    class_index: int


@dataclass
class Materialized:
    samples: dict[str, list[Sample]] = field(default_factory=lambda: {name: [] for name in SPLITS.values()})
    # Rows left out: unknown split, unlabelled or unknown class, or a clip that could not be downloaded.
    skipped: int = 0


def split_s3_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def materialize(
    df: pd.DataFrame,
    dest: str,
    *,
    clip_column: str,
    label_column: str,
    split_column: str,
    item_id_column: str,
    class_names: list[str],
    progress: Callable[[], None] | None = None,
) -> Materialized:
    """Download every usable row's clip into `dest`. `progress` runs after each download, in the calling thread, so
    it may raise (an abort) to stop the rest."""
    index = {name: i for i, name in enumerate(class_names)}
    result = Materialized()
    jobs: list[tuple[str, str, str, str, int]] = []
    columns = [clip_column, label_column, split_column, item_id_column]
    # `itertuples` would rename `class` (a keyword) and `_theseus_item_id` (leading underscore) positionally.
    for values in df[columns].to_dict("records"):
        uri, label, split = values[clip_column], values[label_column], values[split_column]
        split_key = SPLITS.get(split) if isinstance(split, str) else None
        class_index = index.get(label) if isinstance(label, str) else None
        if not isinstance(uri, str) or split_key is None or class_index is None:
            result.skipped += 1
            continue
        bucket, key = split_s3_uri(uri)
        ext = os.path.splitext(key)[1].lower() or ".mp4"
        jobs.append(
            (bucket, key, os.path.join(dest, split_key, f"{values[item_id_column]}{ext}"), split_key, class_index)
        )

    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    try:
        futures = {pool.submit(storage.download_file, b, k, local): (k, local, s, c) for b, k, local, s, c in jobs}
        for future in as_completed(futures):
            key, local, split_key, class_index = futures[future]
            try:
                future.result()
                result.samples[split_key].append(Sample(local, class_index))
            except Exception:
                logger.warning("Could not download %s for training", key, exc_info=True)
                result.skipped += 1
            if progress is not None:
                progress()
    finally:
        # On an abort (or any error) in `progress`, don't wait for the downloads still queued.
        pool.shutdown(wait=True, cancel_futures=True)

    for split_key in ("train", "val"):
        # as_completed yields in finish order; sort so a run does not depend on download timing.
        result.samples[split_key].sort(key=lambda s: s.path)
        if not result.samples[split_key]:
            raise ValueError(
                f"The {'validation' if split_key == 'val' else 'train'} split has no usable clips, which training "
                "needs. Check the snapshot's split settings and that its clips are labelled."
            )
    result.samples["test"].sort(key=lambda s: s.path)
    if result.skipped:
        logger.warning("Left out %d of %d snapshot rows while preparing the video dataset", result.skipped, len(df))
    return result
