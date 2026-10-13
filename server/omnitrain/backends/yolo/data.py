"""Turn a snapshot's dataset.parquet into the local folder tree Ultralytics trains on.

Ludwig reads the parquet and every `s3://` image straight through s3fs. Ultralytics only reads local
files, so each image is downloaded first. Imports pandas only: no ultralytics, no torch.

Image classification layout (docs.ultralytics.com/datasets/classify):

    dest/{train,val,test}/<class dir>/<item id><ext>

Class folders are named by zero-padded INDEX into the snapshot's class list, not by class name. Class names
are free text (slashes, dots, unicode), which a folder name can't carry safely, and Ultralytics takes
`model.names` from the sorted folder names of the train split. With zero-padded indices the sort order is the
index order and `model.names` maps back to the real class names exactly (see `train.py`). Every class folder
exists in every split, even when empty: each split derives its own class indices from its own folders, so a
folder missing from `val` would shift every later class.

Object detection layout (docs.ultralytics.com/datasets/detect): images and same-named label files in parallel
trees, plus a `data.yaml`. Ultralytics finds an image's labels by swapping `/images/` for `/labels/` in its path:

    dest/images/{train,val,test}/<item id><ext>
    dest/labels/{train,val,test}/<item id>.txt      one `class cx cy w h` line per box, all 0..1
    dest/data.yaml                                  `names` maps index -> the REAL class name

Here the class names go into `data.yaml` as they are (it is YAML, so any text is safe), which makes
`model.names` exactly the class list, in index order. An image with no objects gets an empty label file, which
Ultralytics trains on as a background image; an image nobody labelled has no `boxes` value and is left out.
"""

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml

from omnitrain.domain.boxes import parse_boxes
from omnitrain.services import storage

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
    # Detection only: boxes written per split folder, and the data.yaml to hand `YOLO.train(data=...)`.
    box_counts: dict[str, int] = field(default_factory=dict)
    data_yaml: str | None = None


def class_dir_names(class_count: int) -> list[str]:
    """Folder name per class index, zero-padded so that sorting the names sorts the indices."""
    width = len(str(max(class_count - 1, 0)))
    return [f"{i:0{width}d}" for i in range(class_count)]


def split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key` -> (bucket, key)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _download_all(
    jobs: Iterable[tuple[str, str, str, str]], result: MaterializedDataset, progress: Callable[[], None] | None
) -> None:
    """Fetch (bucket, key, local path, split folder) jobs on a small pool, counting what arrived. `progress` runs in
    the calling thread after each one, so it may raise (an abort) to stop the rest."""
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


def _require_train_and_validation(result: MaterializedDataset, row_count: int) -> None:
    for split_dir in ("train", "val"):
        if result.counts[split_dir] == 0:
            raise ValueError(
                f"The {'validation' if split_dir == 'val' else 'train'} split has no usable images, "
                "which YOLO needs to train. Check the snapshot's split settings and that its items are labelled."
            )
    if result.skipped:
        logger.warning("Left out %d of %d snapshot rows while preparing the YOLO dataset", result.skipped, row_count)


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

    _download_all(jobs, result, progress)
    _require_train_and_validation(result, len(df))
    return result


def label_lines(boxes: list[dict[str, Any]], index: dict[str, int]) -> list[str] | None:
    """YOLO label-file lines (`class cx cy w h`, 0..1) for one image's boxes, or None if a box names a class that is
    not in `index`. Boxes are stored as the top-left corner and the size; YOLO wants the centre."""
    lines = []
    for box in boxes:
        class_index = index.get(box["label"])
        if class_index is None:
            return None
        x, y, w, h = (min(max(float(box[k]), 0.0), 1.0) for k in ("x", "y", "w", "h"))
        lines.append(f"{class_index} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
    return lines


def materialize_detection(
    df: pd.DataFrame,
    dest: str,
    *,
    image_column: str,
    boxes_column: str,
    split_column: str,
    item_id_column: str,
    class_names: list[str],
    progress: Callable[[], None] | None = None,
    evaluation_only: bool = False,
) -> MaterializedDataset:
    """Download every labelled row's image into `dest/images/...`, write its label file, and write `data.yaml`.

    `evaluation_only` is for scoring a trained model: every row goes into `val` whatever its split (the caller has
    already chosen which rows to evaluate), and `train` just points at the same images, because Ultralytics insists
    the key exists even though validation never reads it.
    """
    index = {name: i for i, name in enumerate(class_names)}
    for split_dir in SPLIT_DIRS.values():
        os.makedirs(os.path.join(dest, "images", split_dir), exist_ok=True)
        os.makedirs(os.path.join(dest, "labels", split_dir), exist_ok=True)

    result = MaterializedDataset(
        root=dest,
        counts=dict.fromkeys(SPLIT_DIRS.values(), 0),
        box_counts=dict.fromkeys(SPLIT_DIRS.values(), 0),
    )
    jobs: list[tuple[str, str, str, str]] = []
    columns = [image_column, boxes_column, split_column, item_id_column]
    for values in df[columns].to_dict("records"):
        uri, cell, split = values[image_column], values[boxes_column], values[split_column]
        split_dir = ("val" if evaluation_only else SPLIT_DIRS.get(split)) if isinstance(split, str) else None
        boxes = parse_boxes(cell)
        lines = label_lines(boxes, index) if boxes is not None else None
        if not isinstance(uri, str) or split_dir is None or lines is None:
            # No split, an unlabelled image (no `boxes` value), or a box of a class the snapshot does not list.
            result.skipped += 1
            continue
        bucket, key = split_s3_uri(uri)
        stem = str(values[item_id_column])
        ext = os.path.splitext(key)[1].lower() or ".jpg"
        with open(os.path.join(dest, "labels", split_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result.box_counts[split_dir] += len(lines)
        jobs.append((bucket, key, os.path.join(dest, "images", split_dir, f"{stem}{ext}"), split_dir))

    _download_all(jobs, result, progress)
    if evaluation_only:
        if result.counts["val"] == 0:
            raise ValueError("None of the evaluation images could be prepared")
    else:
        _require_train_and_validation(result, len(df))
        if result.box_counts["train"] == 0:
            raise ValueError(
                "The train split has images but no boxes, so there is nothing for a detector to learn. "
                "Label some objects, or check the snapshot's split settings."
            )

    result.data_yaml = os.path.join(dest, "data.yaml")
    config: dict[str, Any] = {
        "path": os.path.abspath(dest),
        "train": "images/val" if evaluation_only else "images/train",
        "val": "images/val",
        "names": dict(enumerate(class_names)),
    }
    if result.counts["test"]:
        config["test"] = "images/test"
    with open(result.data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return result
