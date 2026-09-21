"""Regression tests for the golden-sample builder.

`_build_golden_sample` writes `expected.json`, which every devkit/app bundle's
verify script checks its re-implemented preprocessing against. It shipped
broken — `rows.iloc[[0]].to_frame()` raised AttributeError on every call
(`.to_frame()` is a Series method; `.iloc[[0]]` already yields a DataFrame) and
the broad `except Exception` swallowed it into `return None`, so no export ever
produced the file and every bundle silently lost its Verify section.

These tests pin the row-selection shape without needing a real Ludwig model.
"""

import numpy as np
import pandas as pd
import pytest

from theseus.constants import SPLIT_COLUMN_NAME
from theseus.jobs.export import _shape_input_value


def _select_sample(df: pd.DataFrame) -> pd.DataFrame:
    """The row-selection prelude of `_build_golden_sample`, verbatim."""
    rows = df[df[SPLIT_COLUMN_NAME] == "test"] if SPLIT_COLUMN_NAME in df.columns else df
    if len(rows) == 0:
        rows = df
    return rows.iloc[[0]]


def test_sample_selection_yields_a_single_row_dataframe():
    df = pd.DataFrame(
        {
            "image_path": ["s3://a.jpg", "s3://b.jpg", "s3://c.jpg"],
            "class": ["cat", "dog", "cat"],
            SPLIT_COLUMN_NAME: ["train", "test", "train"],
        }
    )

    sample = _select_sample(df)

    # A DataFrame, not a Series — LudwigModel.predict(dataset=...) requires it.
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 1
    assert sample.iloc[0]["image_path"] == "s3://b.jpg"


def test_falls_back_to_full_frame_when_no_test_split():
    df = pd.DataFrame(
        {
            "text": ["hello", "world"],
            "class": ["greeting", "noun"],
            SPLIT_COLUMN_NAME: ["train", "validation"],
        }
    )

    sample = _select_sample(df)

    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 1
    assert sample.iloc[0]["text"] == "hello"


def test_handles_frame_with_no_split_column():
    df = pd.DataFrame({"text": ["only"], "class": ["x"]})

    sample = _select_sample(df)

    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 1


def test_to_frame_on_a_dataframe_is_the_original_bug():
    """Documents why this file exists: the old code called a Series method."""
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(AttributeError):
        df.iloc[[0]].to_frame()  # type: ignore[operator]


# ──────────────────────────────────────────────────────────────────
# _shape_input_value — single-feature scalar vs. multi-feature (tabular) record
# ──────────────────────────────────────────────────────────────────


class _FakeFeature:
    def __init__(self, column: str):
        self.column = column


def test_single_input_feature_keeps_the_scalar_shape():
    row = pd.Series({"image_path": "s3://bucket/cat.jpg", "class": "cat"})

    column, value = _shape_input_value([_FakeFeature("image_path")], row)

    assert column == "image_path"
    assert value == "s3://bucket/cat.jpg"


def test_multiple_input_features_become_a_record_with_no_single_column():
    row = pd.Series({"age": 34.0, "income": 52000.0, "class": "approved"})

    column, value = _shape_input_value([_FakeFeature("age"), _FakeFeature("income")], row)

    assert column is None
    assert value == {"age": 34.0, "income": 52000.0}


def test_numpy_scalars_are_converted_to_plain_python_types():
    # A parquet-backed DataFrame column yields numpy scalars, not plain
    # Python ones — json.dumps can't serialize those directly (see
    # services/storage.py's upload_json, which would otherwise silently
    # stringify them via its `default=str` fallback instead of writing a
    # real number).
    row = pd.Series({"age": np.float64(34.0)})

    _, value = _shape_input_value([_FakeFeature("age")], row)

    assert isinstance(value, float)
    assert not isinstance(value, np.floating)
