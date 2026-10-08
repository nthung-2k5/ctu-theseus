"""Bounding-box annotations: validation, the "no objects" marker and the snapshot's `boxes` cell."""

import io
import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from theseus.services import snapshot as snap
from theseus.services.boxes import BoundingBox, boxes_json, is_background, parse_boxes, validate_box_annotation
from theseus.services.task_registry import ColumnSpec, get_task_descriptor

CAT, DOG, GONE = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
NAMES = {CAT: "cat", DOG: "dog"}  # GONE is a deleted class: not among the active ones
BOX = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
MARKER = {"background": True}


class TestValidation:
    def test_a_box_inside_the_image_with_a_class_is_valid(self):
        validate_box_annotation(CAT, BOX)

    def test_a_box_may_touch_every_edge(self):
        validate_box_annotation(CAT, {"x": 0, "y": 0, "w": 1, "h": 1})
        # Floating point: 0.7 + 0.3 is not exactly 1.
        validate_box_annotation(CAT, {"x": 0.7, "y": 0.1 + 0.2, "w": 0.3, "h": 0.7})

    @pytest.mark.parametrize(
        "label",
        [
            {"x": -0.1, "y": 0, "w": 0.5, "h": 0.5},  # off the left edge
            {"x": 1.0, "y": 0, "w": 0.1, "h": 0.5},  # starts on the right edge
            {"x": 0.6, "y": 0, "w": 0.5, "h": 0.5},  # runs past the right edge
            {"x": 0, "y": 0.6, "w": 0.5, "h": 0.5},  # runs past the bottom edge
            {"x": 0.1, "y": 0.1, "w": 0, "h": 0.5},  # no width
            {"x": 0.1, "y": 0.1, "w": 0.5, "h": -0.2},  # negative height
            {"x": 0.1, "y": 0.1, "w": 0.5},  # missing h
            {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "angle": 3},  # unknown key
            {"x": "left", "y": 0.1, "w": 0.5, "h": 0.5},
            {"x": float("nan"), "y": 0.1, "w": 0.5, "h": 0.5},
            None,
            [0.1, 0.1, 0.5, 0.5],
        ],
    )
    def test_a_malformed_box_is_refused_with_a_message(self, label):
        with pytest.raises(ValueError, match="bounding box|edge"):
            validate_box_annotation(CAT, label)

    def test_a_box_needs_a_class(self):
        with pytest.raises(ValueError, match="needs a class"):
            validate_box_annotation(None, BOX)

    def test_a_marker_has_no_class_and_nothing_else(self):
        validate_box_annotation(None, MARKER)
        with pytest.raises(ValueError, match="no class"):
            validate_box_annotation(CAT, MARKER)
        with pytest.raises(ValueError):
            validate_box_annotation(None, {"background": True, "x": 0.1})
        with pytest.raises(ValueError):
            validate_box_annotation(None, {"background": False})

    def test_is_background_only_for_the_exact_marker(self):
        assert is_background(MARKER)
        assert not is_background({"background": False})
        assert not is_background({"background": 1})  # truthy but not the marker
        assert not is_background(BOX)
        assert not is_background(None)

    def test_the_model_rejects_what_the_validator_rejects(self):
        assert BoundingBox.model_validate(BOX).w == 0.3
        with pytest.raises(ValueError):
            BoundingBox.model_validate({"x": 0.9, "y": 0, "w": 0.5, "h": 0.5})


class TestBoxesJson:
    def test_boxes_are_labelled_by_class_name_in_the_order_given(self):
        cell = boxes_json([(CAT, BOX), (DOG, {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2})], NAMES)
        assert json.loads(cell) == [{"label": "cat", **BOX}, {"label": "dog", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}]

    def test_a_marker_means_labelled_and_empty(self):
        assert boxes_json([(None, MARKER)], NAMES) == "[]"

    def test_no_annotation_at_all_means_not_labelled(self):
        # None, not "[]": an unlabelled image must never be taught to the model as "nothing here".
        assert boxes_json([], NAMES) is None

    def test_boxes_win_over_a_stray_marker(self):
        assert json.loads(boxes_json([(None, MARKER), (CAT, BOX)], NAMES)) == [{"label": "cat", **BOX}]

    def test_a_box_of_a_deleted_class_is_dropped_and_may_leave_the_item_unlabelled(self):
        assert json.loads(boxes_json([(GONE, BOX), (CAT, BOX)], NAMES)) == [{"label": "cat", **BOX}]
        assert boxes_json([(GONE, BOX)], NAMES) is None

    def test_round_trips_through_parse_boxes(self):
        cell = boxes_json([(CAT, BOX)], NAMES)
        assert parse_boxes(cell) == [{"label": "cat", **BOX}]
        assert parse_boxes("[]") == []
        assert parse_boxes(None) is None


class TestSnapshotColumn:
    COLUMN = ColumnSpec(name="boxes", kind="boxes")

    @staticmethod
    def member(*annotations: snap.AnnotationView) -> snap.SnapshotMember:
        return snap.SnapshotMember(uuid.uuid4(), "train", "pool/a.png", annotations=list(annotations))

    def test_resolves_only_bounding_box_annotations(self):
        m = self.member(
            snap.AnnotationView(CAT, "bounding_box", BOX),
            snap.AnnotationView(CAT, "classification"),  # a stray annotation of another type is ignored
        )
        assert json.loads(snap.resolve_column_value(self.COLUMN, m, NAMES)) == [{"label": "cat", **BOX}]

    def test_unlabelled_marker_and_boxed_items(self):
        assert snap.resolve_column_value(self.COLUMN, self.member(), NAMES) is None
        assert (
            snap.resolve_column_value(
                self.COLUMN, self.member(snap.AnnotationView(None, "bounding_box", MARKER)), NAMES
            )
            == "[]"
        )

    def test_the_detection_task_builds_rows_and_a_string_parquet_column(self):
        task = get_task_descriptor("object_detection")
        columns = snap.derive_columns(task, [])
        assert [c.name for c in columns] == ["image_path", "boxes", "split", "_ludwig_split_idx", "_theseus_item_id"]

        rows = snap.build_rows(columns, [self.member(snap.AnnotationView(CAT, "bounding_box", BOX))], NAMES)
        assert rows[0]["image_path"] == "s3://theseus-datasets/pool/a.png"
        assert json.loads(rows[0]["boxes"])[0]["label"] == "cat"

        parquet = pq.read_table(io.BytesIO(snap.to_parquet_bytes(task, columns, rows)))
        assert parquet.schema.field("boxes").type == pa.string()
        assert parquet.column("boxes").to_pylist() == [rows[0]["boxes"]]

    def test_no_class_counts_for_detection(self):
        # `class_counts` feeds class weights for classification; a detection snapshot has no label column.
        task = get_task_descriptor("object_detection")
        columns = snap.derive_columns(task, [])
        assert snap.class_counts(task, columns, []) is None
