import os

import pandas as pd
import pytest

from theseus.backends.yolo import data
from theseus.jobs import abort


def row(item_id, label, split, uri=None):
    return {
        "image_path": uri if uri is not None else f"s3://theseus-datasets/pool/{item_id}.png",
        "class": label,
        "split": split,
        "_theseus_item_id": item_id,
    }


def materialize(df, dest, class_names=("cat", "dog"), **kw):
    return data.materialize_classification(
        df,
        str(dest),
        image_column="image_path",
        label_column="class",
        split_column="split",
        item_id_column="_theseus_item_id",
        class_names=list(class_names),
        **kw,
    )


@pytest.fixture
def downloads(monkeypatch):
    fetched: list[tuple[str, str]] = []

    def fake_download(bucket, key, local_path):
        fetched.append((bucket, key))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"png")
        return local_path

    monkeypatch.setattr(data.storage, "download_file", fake_download)
    return fetched


def files(root) -> list[str]:
    return sorted(
        os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for d, _, fs in os.walk(root) for f in fs
    )


def test_images_land_under_split_and_class_index_folders(tmp_path, downloads):
    df = pd.DataFrame(
        [
            row("a1", "cat", "train"),
            row("a2", "dog", "train"),
            row("a3", "cat", "validation"),
            row("a4", "dog", "test"),
        ]
    )
    result = materialize(df, tmp_path)

    assert files(tmp_path) == ["test/1/a4.png", "train/0/a1.png", "train/1/a2.png", "val/0/a3.png"]
    assert result.counts == {"train": 2, "val": 1, "test": 1}
    assert result.skipped == 0
    assert ("theseus-datasets", "pool/a1.png") in downloads


def test_every_class_folder_exists_in_every_split_even_when_empty(tmp_path, downloads):
    # Ultralytics derives each split's class indices from that split's own folders, so a missing folder
    # (here: "dog" has no validation images, and no test images at all) would shift every later class.
    df = pd.DataFrame([row("a1", "cat", "train"), row("a2", "dog", "train"), row("a3", "cat", "validation")])
    materialize(df, tmp_path)

    for split in ("train", "val", "test"):
        assert sorted(os.listdir(tmp_path / split)) == ["0", "1"]


def test_class_folders_are_zero_padded_so_sorting_them_sorts_the_indices(tmp_path, downloads):
    assert data.class_dir_names(2) == ["0", "1"]
    names = data.class_dir_names(12)
    assert names[:3] == ["00", "01", "02"]
    assert sorted(names) == names  # string sort order is index order, which is how Ultralytics orders model.names


def test_class_names_never_reach_the_file_system(tmp_path, downloads):
    df = pd.DataFrame([row("a1", "a/b: c?", "train"), row("a2", "x", "validation")])
    materialize(df, tmp_path, class_names=("a/b: c?", "x"))
    assert files(tmp_path) == ["train/0/a1.png", "val/1/a2.png"]


def test_unusable_rows_are_skipped_and_counted(tmp_path, downloads):
    df = pd.DataFrame(
        [
            row("ok", "cat", "train"),
            row("ok2", "cat", "validation"),
            row("nolabel", None, "train"),
            row("unknown", "fish", "train"),
            row("nosplit", "cat", "holdout"),
            row("nouri", "cat", "train", uri=None) | {"image_path": None},
        ]
    )
    result = materialize(df, tmp_path)
    assert result.skipped == 4
    assert result.counts["train"] == 1


def test_a_failed_download_skips_that_image_only(tmp_path, monkeypatch):
    def flaky(bucket, key, local_path):
        if key.endswith("bad.png"):
            raise OSError("404")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        open(local_path, "wb").close()
        return local_path

    monkeypatch.setattr(data.storage, "download_file", flaky)
    df = pd.DataFrame([row("good", "cat", "train"), row("bad", "cat", "train"), row("v", "dog", "validation")])
    result = materialize(df, tmp_path)
    assert result.counts["train"] == 1
    assert result.skipped == 1


@pytest.mark.parametrize("missing", ["train", "validation"])
def test_train_and_validation_must_both_have_images(tmp_path, downloads, missing):
    present = "validation" if missing == "train" else "train"
    with pytest.raises(ValueError, match="no usable images"):
        materialize(pd.DataFrame([row("a", "cat", present)]), tmp_path)


def test_progress_runs_per_image_and_can_abort(tmp_path, downloads):
    df = pd.DataFrame([row(f"t{i}", "cat", "train") for i in range(5)] + [row("v", "dog", "validation")])
    seen = {"n": 0}

    def progress():
        seen["n"] += 1
        if seen["n"] == 2:
            raise abort.TrainingAborted("stop")

    with pytest.raises(abort.TrainingAborted):
        materialize(df, tmp_path, progress=progress)
    assert seen["n"] == 2


def test_uri_without_extension_defaults_to_jpg(tmp_path, downloads):
    df = pd.DataFrame([row("a", "cat", "train", uri="s3://b/pool/blob"), row("v", "cat", "validation")])
    materialize(df, tmp_path)
    assert "train/0/a.jpg" in files(tmp_path)


def test_split_s3_uri():
    assert data.split_s3_uri("s3://theseus-datasets/pool/p/abc.png") == ("theseus-datasets", "pool/p/abc.png")
