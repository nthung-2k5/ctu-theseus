"""Turn a snapshot's dataset.parquet into the local folder tree Ultralytics trains on.

Ludwig reads the parquet and every `s3://` image straight through s3fs. Ultralytics only reads local
files, so each image is downloaded first. Imports pandas only: no ultralytics, no torch.

Image classification layout (`yolo.ultralytics.com/datasets/classify`):

    dest/{train,val,test}/<class dir>/<item id><ext>

Class folders are named by zero-padded INDEX into the snapshot's class list, not by class name. Class names
are free text (slashes, dots, unicode), which a folder name can't carry safely, and Ultralytics takes
`model.names` from the sorted folder names of the train split. With zero-padded indices the sort order is the
index order and `model.names` maps back to the real class names exactly (see `train.py`). Every class folder
exists in every split, even when empty: each split derives its own class indices from its own folders, so a
folder missing from `val` would shift every later class.
"""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from theseus.services import storage

logger = logging.getLogger(__name__)

# Snapshot split name -> Ultralytics split folder.
SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}

DOWNLOAD_WORKERS = 8


@dataclass
class MaterializedDataset:
    root: str
    # Images written per Ultralytics split folder.
    counts: dict[str, int] = field(default_factory=dict)
    # Rows left out: unknown split, unlabelled or unknown class, or an image that could not be downloaded.
    skipped: int = 0


def class_dir_names(class_count: int) -> list[str]:
    """Folder name per class index, zero-padded so that sorting the names sorts the indices."""
    width = len(str(max(class_count - 1, 0)))
    return [f"{i:0{width}d}" for i in range(class_count)]


def split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key` -> (bucket, key)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def materialize_classification(
    df: pd.DataFrame,
    dest: str,
    *,
    image_column: str,
    label_column: str,
    split_column: str,
    item_id_column: str,
    class_names: list[str],
    progress: Callable[[], None] | None = None,
) -> MaterializedDataset:
    """Download every usable row's image into `dest`. `progress` runs after each download, in the calling
    thread, so it may raise (an abort) to stop the rest."""
    dir_names = class_dir_names(len(class_names))
    index = {name: i for i, name in enumerate(class_names)}
    for split_dir in SPLIT_DIRS.values():
        for name in dir_names:
            os.makedirs(os.path.join(dest, split_dir, name), exist_ok=True)

    result = MaterializedDataset(root=dest, counts=dict.fromkeys(SPLIT_DIRS.values(), 0))
    jobs: list[tuple[str, str, str, str]] = []
    columns = [image_column, label_column, split_column, item_id_column]
    # `itertuples` would rename `class` (a keyword) and `_theseus_item_id` (leading underscore) positionally.
    for values in df[columns].to_dict("records"):
        uri, label, split = values[image_column], values[label_column], values[split_column]
        split_dir = SPLIT_DIRS.get(split) if isinstance(split, str) else None
        class_index = index.get(label) if isinstance(label, str) else None
        if not isinstance(uri, str) or split_dir is None or class_index is None:
            result.skipped += 1
            continue
        bucket, key = split_s3_uri(uri)
        ext = os.path.splitext(key)[1].lower() or ".jpg"
        local = os.path.join(dest, split_dir, dir_names[class_index], f"{values[item_id_column]}{ext}")
        jobs.append((bucket, key, local, split_dir))

    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    try:
        futures = {
            pool.submit(storage.download_file, bucket, key, local): (key, split_dir)
            for bucket, key, local, split_dir in jobs
        }
        for future in as_completed(futures):
            key, split_dir = futures[future]
            try:
                future.result()
                result.counts[split_dir] += 1
            except Exception:
                logger.warning("Could not download %s for training", key, exc_info=True)
                result.skipped += 1
            if progress is not None:
                progress()
    finally:
        # On an abort (or any error) in `progress`, don't wait for the downloads still queued.
        pool.shutdown(wait=True, cancel_futures=True)

    for split_dir in ("train", "val"):
        if result.counts[split_dir] == 0:
            raise ValueError(
                f"The {'validation' if split_dir == 'val' else 'train'} split has no usable images, "
                "which YOLO needs to train. Check the snapshot's split settings and that its items are labelled."
            )
    if result.skipped:
        logger.warning("Left out %d of %d snapshot rows while preparing the YOLO dataset", result.skipped, len(df))
    return result
