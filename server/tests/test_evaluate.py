import pandas as pd

from theseus.services.evaluate import pick_eval_split

SPLIT_COL = "split"


def test_pick_eval_split_prefers_test_when_present():
    df = pd.DataFrame({SPLIT_COL: ["train", "test", "validation", "test"], "x": [1, 2, 3, 4]})
    rows, split = pick_eval_split(df, SPLIT_COL)
    assert split == "test"
    assert len(rows) == 2


def test_pick_eval_split_falls_back_to_validation_when_test_empty():
    df = pd.DataFrame({SPLIT_COL: ["train", "validation", "train"], "x": [1, 2, 3]})
    rows, split = pick_eval_split(df, SPLIT_COL)
    assert split == "validation"
    assert len(rows) == 1


def test_pick_eval_split_falls_back_to_full_when_no_split_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    rows, split = pick_eval_split(df, SPLIT_COL)
    assert split == "full"
    assert len(rows) == 3


def test_pick_eval_split_falls_back_to_full_when_neither_test_nor_validation_present():
    df = pd.DataFrame({SPLIT_COL: ["train", "train"], "x": [1, 2]})
    rows, split = pick_eval_split(df, SPLIT_COL)
    assert split == "full"
    assert len(rows) == 2
