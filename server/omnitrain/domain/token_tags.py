"""Token tags: one tag per word of a text, the ground truth of `token_classification`.

A text's words are what `str.split()` gives: runs of Unicode whitespace separate them, nothing else does, so "Hanoi,"
is one word. That is the same definition the inference route uses to line a prediction up with its input, and it is
what a tagging dataset in CoNLL style already is (punctuation as a word of its own, separated by a space).

An item's `token_tags` annotation holds `label_structured = ["B-LOC", "O", ...]`, exactly one tag per word, in order.
A tag is a non-empty string with no whitespace in it (a snapshot writes the tags into one space-separated cell, and a
tag that held a space would split in two). The tag names are the dataset's own: `O`, `B-PER` and `I-PER` are a
convention, not something this module knows about.
"""

from typing import Any

MAX_TAG_LENGTH = 64
# The tag a prediction falls back to for a word the model never read (the text was longer than the model's length), if
# the tag set has it.
OUTSIDE = "O"

# The code points (inclusive ranges) `str.split()` separates words on: exactly those `str.isspace()` is true for. All
# are in the Basic Multilingual Plane, so a client that reads UTF-16 code units splits a text the same way as one that
# reads code points. The export manifest hands this list to a client; a test holds it to `str.isspace()`.
WHITESPACE_RANGES: tuple[tuple[int, int], ...] = (
    (0x09, 0x0D), (0x1C, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
    (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
)  # fmt: skip


def words(text: str) -> list[str]:
    return text.split()


def tag_shape_error(tags: Any) -> str | None:
    """Why `tags` cannot be a tag list (None when it can): a non-empty list of strings that are neither empty nor
    contain whitespace. The length against the text is `tag_count_error`'s business."""
    if not isinstance(tags, list) or not tags:
        return "Token tags must be a non-empty list of tags, one per word"
    for tag in tags:
        if not isinstance(tag, str) or not tag or tag != tag.strip() or len(tag.split()) != 1:
            return "A tag must be a non-empty string without spaces"
        if len(tag) > MAX_TAG_LENGTH:
            return f"A tag may have at most {MAX_TAG_LENGTH} characters"
    return None


def tag_count_error(tags: list[str], text: str) -> str | None:
    count = len(words(text))
    if len(tags) != count:
        return (
            f"The text has {count} word{'s' if count != 1 else ''} but {len(tags)} tag{'s' if len(tags) != 1 else ''}"
        )
    return None


def tags_cell(tags: Any) -> str | None:
    """The snapshot cell for an annotation's tags: the tags joined by single spaces, or None when they are not a tag
    list. Splitting the cell on whitespace gives them back."""
    return None if tag_shape_error(tags) else " ".join(tags)


def cell_tags(cell: Any) -> list[str]:
    return cell.split() if isinstance(cell, str) else []


def tag_set(cells: Any) -> list[str]:
    """Every distinct tag in the cells, `O` first and the rest alphabetical: one order for the same data, whatever the
    row order, which the trained model stores as its class order."""
    seen = {tag for cell in cells for tag in cell_tags(cell)}
    return sorted(seen, key=lambda tag: (tag != OUTSIDE, tag))
