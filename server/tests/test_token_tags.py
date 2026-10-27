"""Token tags, the ground truth of token classification: one tag per whitespace-separated word, written by the API,
validated against the item's own text, and written into the snapshot as one space-separated cell."""

import uuid

import pytest

from omnitrain.domain.task_registry import ColumnSpec, get_task_descriptor
from omnitrain.domain.token_tags import (
    cell_tags,
    tag_count_error,
    tag_set,
    tag_shape_error,
    tags_cell,
    words,
)
from omnitrain.services import snapshot as snap


class TestWords:
    def test_words_are_runs_of_non_whitespace_so_punctuation_stays_attached(self):
        assert words("Hanoi, Việt  Nam\t!\n") == ["Hanoi,", "Việt", "Nam", "!"]
        assert words("a b　c") == ["a", "b", "c"]
        assert words("   ") == [] and words("") == []


class TestShape:
    @pytest.mark.parametrize("tags", [["O"], ["B-LOC", "I-LOC", "O"], ["x" * 64]])
    def test_a_list_of_plain_tags_is_a_tag_list(self, tags):
        assert tag_shape_error(tags) is None

    @pytest.mark.parametrize(
        "tags",
        [None, "O O", [], ["O", ""], ["O", " "], ["B LOC"], [" O"], ["O", 1], ["O", None], ["x" * 65], {"O": 1}],
    )
    def test_anything_else_is_not(self, tags):
        assert tag_shape_error(tags)

    def test_the_count_must_match_the_words(self):
        assert tag_count_error(["O", "O"], "two words") is None
        assert "3 words but 2 tags" in tag_count_error(["O", "O"], "three words here")
        assert "1 word but 2 tags" in tag_count_error(["O", "O"], "one")


class TestCell:
    def test_the_cell_is_the_tags_joined_by_single_spaces_and_splits_back(self):
        cell = tags_cell(["B-PER", "I-PER", "O"])
        assert cell == "B-PER I-PER O" and cell_tags(cell) == ["B-PER", "I-PER", "O"]

    def test_something_that_is_not_a_tag_list_has_no_cell(self):
        assert tags_cell(None) is None and tags_cell(["a b"]) is None and tags_cell([]) is None
        assert cell_tags(None) == [] and cell_tags("") == []

    def test_the_tag_set_puts_outside_first_and_is_the_same_whatever_the_row_order(self):
        rows = ["B-PER O", "O B-LOC I-LOC", None, "I-PER"]
        assert tag_set(rows) == ["O", "B-LOC", "B-PER", "I-LOC", "I-PER"]
        assert tag_set(reversed(rows)) == tag_set(rows)
        assert tag_set([]) == []


TASK = get_task_descriptor("token_classification")
ITEM = uuid.uuid4()


def member(tags, text="Anna lives in Hue"):
    return snap.SnapshotMember(
        item_id=ITEM, split_type="train", raw_text=text, annotations=[snap.AnnotationView(None, "token_tags", tags)]
    )


class TestSnapshot:
    def label(self, m):
        return snap.resolve_column_value(ColumnSpec(name="tags", kind="label"), m, {})

    def test_the_label_cell_holds_the_tags(self):
        assert self.label(member(["B-PER", "O", "O", "B-LOC"])) == "B-PER O O B-LOC"

    def test_tags_that_do_not_fit_the_text_are_no_label_at_all(self):
        assert self.label(member(["B-PER", "O"])) is None
        assert self.label(member("not a list")) is None

    def test_an_item_with_no_tags_has_no_label(self):
        assert self.label(snap.SnapshotMember(item_id=ITEM, split_type="train", raw_text="a b")) is None

    def test_a_cell_has_no_class_counts(self):
        cols = [ColumnSpec(name="tags", kind="label")]
        assert snap.class_counts(TASK, cols, [{"tags": "O O B-LOC"}, {"tags": "O"}]) is None

    def test_the_snapshot_columns_are_text_tags_and_the_split(self):
        names = [c.name for c in snap.derive_columns(TASK, [member(["O"] * 4)])]
        assert names[:3] == ["text", "tags", "split"]


# -- Over HTTP ----------------------------------------------------------------------------------------------------


async def project(c):
    r = await c.post("/api/projects", json={"name": "p", "description": None, "task": "token_classification"})
    assert r.status_code == 200, r.text
    return r.json()["project"]["id"]


async def add_text(c, pid, text, split="train"):
    r = await c.post(
        f"/api/projects/{pid}/items", json={"items": [{"split": split, "textFeatures": {"rawText": text}}]}
    )
    assert r.status_code == 200, r.text
    return r.json()["created"][0]["id"]


async def tag(c, item_id, tags):
    return await c.post(
        f"/api/items/{item_id}/annotations", json={"annotationType": "token_tags", "labelStructured": tags}
    )


async def test_tags_are_stored_on_a_text_item_with_one_tag_per_word(client, new_user):
    c = await new_user()
    item = await add_text(c, await project(c), "Anna lives in Hue")
    r = await tag(c, item, ["B-PER", "O", "O", "B-LOC"])
    assert r.status_code == 200, r.text
    assert r.json()["annotation"]["labelStructured"] == ["B-PER", "O", "O", "B-LOC"]


async def test_a_tag_list_of_the_wrong_length_is_refused_with_both_counts(client, new_user):
    c = await new_user()
    item = await add_text(c, await project(c), "Anna lives in Hue")
    r = await tag(c, item, ["B-PER", "O"])
    assert r.status_code == 422 and "4 words but 2 tags" in r.text
    assert (await c.get(f"/api/items/{item}/annotations")).json()["annotations"] == []


@pytest.mark.parametrize("tags", [[], ["B LOC", "O", "O", "O"], "O O O O", None])
async def test_something_that_is_not_a_tag_list_is_refused(client, new_user, tags):
    c = await new_user()
    item = await add_text(c, await project(c), "Anna lives in Hue")
    assert (await tag(c, item, tags)).status_code == 422


async def test_tags_can_be_replaced_but_not_by_a_list_that_does_not_fit(client, new_user):
    c = await new_user()
    item = await add_text(c, await project(c), "Anna lives in Hue")
    created = (await tag(c, item, ["O", "O", "O", "O"])).json()["annotation"]["id"]

    r = await c.patch(f"/api/annotations/{created}", json={"labelStructured": ["B-PER", "O", "O", "B-LOC"]})
    assert r.status_code == 200, r.text
    assert r.json()["annotation"]["labelStructured"] == ["B-PER", "O", "O", "B-LOC"]

    r = await c.patch(f"/api/annotations/{created}", json={"labelStructured": ["O"]})
    assert r.status_code == 422 and "4 words but 1 tag" in r.text
    stored = (await c.get(f"/api/items/{item}/annotations")).json()["annotations"][0]["labelStructured"]
    assert stored == ["B-PER", "O", "O", "B-LOC"]


def test_the_whitespace_ranges_are_exactly_what_str_split_separates_on():
    from omnitrain.domain.token_tags import WHITESPACE_RANGES, words

    listed = {c for low, high in WHITESPACE_RANGES for c in range(low, high + 1)}
    assert listed == {c for c in range(0x110000) if chr(c).isspace()}
    assert max(high for _, high in WHITESPACE_RANGES) < 0x10000  # a client reading UTF-16 splits the same way
    # And a text really is cut at each of them, with nothing else.
    for code in sorted(listed):
        assert words(f"a{chr(code)}b") == ["a", "b"]
    assert words("a​b,c") == ["a​b,c"]  # a zero-width space is not whitespace
