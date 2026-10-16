import json

import pytest

from omnitrain.domain.task_registry import (
    TASK_REGISTRY,
    find_task_descriptor,
    get_inference_input_spec,
    get_inference_output_kind,
    get_task_descriptor,
    is_classification_task,
    list_selectable_tasks,
    registry_json,
    task_to_modality,
)

BUILTIN_TASKS = [
    "image_classification", "text_classification", "tabular_classification", "tabular_regression",
    "audio_classification", "token_classification", "text_generation", "summarization",
    "sequence_to_sequence", "question_answering", "image_captioning", "audio_captioning",
    "automatic_speech_recognition", "video_classification", "object_detection",
    "image_segmentation", "audio_segmentation", "tabular_clustering", "tabular_anomaly_detection",
    "text_embedding",
]  # fmt: skip


def test_the_builtin_tasks_are_registered_in_display_order():
    # The order is what the web's task picker shows within a modality; it is pinned by each task's `order`.
    assert list(TASK_REGISTRY) == BUILTIN_TASKS


def test_unknown_task_is_a_key_error_or_none():
    with pytest.raises(KeyError):
        get_task_descriptor("no_such_task")
    assert find_task_descriptor("no_such_task") is None
    assert "no_such_task" not in TASK_REGISTRY


def test_fifteen_selectable_tasks_and_five_planned():
    assert len(list_selectable_tasks()) == 15
    planned = [t for t in TASK_REGISTRY.values() if t.status == "planned"]
    assert len(planned) == 5
    assert all(t.output is None and t.columns == [] for t in planned)


def test_classification_helpers():
    assert is_classification_task("image_classification")
    assert not is_classification_task("tabular_regression")
    assert not is_classification_task(None)
    assert task_to_modality("audio_classification") == "audio"


class TestInferenceInputSpec:
    def test_file_tasks_report_accepted_mime_types(self):
        assert get_inference_input_spec("image_classification") == {
            "kind": "file",
            "accept": ["image/jpeg", "image/png"],
        }

    def test_record_tasks_carry_no_fixed_field_list(self):
        assert get_inference_input_spec("tabular_classification") == {"kind": "record"}
        assert get_inference_input_spec("tabular_regression") == {"kind": "record"}

    def test_single_input_text_tasks_report_one_field_named_after_their_column(self):
        assert get_inference_input_spec("text_classification") == {"kind": "text", "fields": ["text"]}
        assert get_inference_input_spec("text_generation") == {"kind": "text", "fields": ["prompt"]}
        assert get_inference_input_spec("summarization") == {"kind": "text", "fields": ["document"]}
        assert get_inference_input_spec("sequence_to_sequence") == {"kind": "text", "fields": ["source"]}

    def test_question_answering_reports_both_input_fields(self):
        assert get_inference_input_spec("question_answering") == {"kind": "text", "fields": ["context", "question"]}

    def test_unsupported_task_has_no_spec(self):
        # planned tasks default to a file payload, but have no output kind to derive text fields from
        assert get_inference_input_spec("object_detection")["kind"] == "file"


class TestInferenceOutputKind:
    def test_category_outputs_are_classification(self):
        for t in ("image_classification", "text_classification", "tabular_classification"):
            assert get_inference_output_kind(t) == "classification"

    def test_number_outputs_are_regression(self):
        assert get_inference_output_kind("tabular_regression") == "regression"

    def test_sequence_outputs_are_tokens(self):
        assert get_inference_output_kind("token_classification") == "tokens"

    def test_generated_text_outputs_are_text(self):
        for t in ("text_generation", "summarization", "sequence_to_sequence", "question_answering", "image_captioning"):
            assert get_inference_output_kind(t) == "text"

    def test_unsupported_task_raises(self):
        with pytest.raises(ValueError):
            get_inference_output_kind("image_segmentation")  # planned: no output shape yet


def test_descriptors_are_immutable_data_no_backend_can_corrupt():
    # Compiling deep-copies feature dicts, so the shared registry must be unchanged afterwards.
    before = get_task_descriptor("image_classification").model_dump()
    from omnitrain.domain.task_registry import SnapshotContext
    from omnitrain.plugins.backends.ludwig.compile import LudwigHyperparameters, compile_ludwig_config

    compile_ludwig_config(
        get_task_descriptor("image_classification"),
        SnapshotContext(),
        LudwigHyperparameters(image_size=64),
    )
    assert get_task_descriptor("image_classification").model_dump() == before


def test_registry_json_is_camelcase_json_with_pre_evaluated_inference_values():
    data = registry_json()
    json.dumps(data)  # must be serializable
    qa = data["tasks"]["question_answering"]
    assert qa["inferenceInputSpec"] == {"kind": "text", "fields": ["context", "question"]}
    assert qa["inferenceOutputKind"] == "text"
    assert qa["itemSpec"]["payload"] == "inline_text"
    assert qa["annotation"]["requiresLabelClasses"] is False
    assert qa["output"] == {"column": "answer", "kind": "text"}
    assert "inferenceInputSpec" not in data["tasks"]["image_segmentation"]  # planned: nothing to pre-evaluate
    assert "output" not in data["tasks"]["image_segmentation"]
    detection = data["tasks"]["object_detection"]
    assert detection["inferenceOutputKind"] == "detections"
    assert detection["inferenceInputSpec"]["kind"] == "file"
    assert [c["kind"] for c in detection["columns"]] == ["storage_uri", "boxes", "split"]


class TestObjectDetection:
    def test_is_a_real_task_whose_output_is_boxes(self):
        d = get_task_descriptor("object_detection")
        assert d.status == "experimental"
        assert (d.annotation.type, d.annotation.requires_label_classes) == ("bounding_box", True)
        assert [(c.name, c.kind) for c in d.columns] == [
            ("image_path", "storage_uri"),
            ("boxes", "boxes"),
            ("split", "split"),
        ]
        assert d.output is not None and (d.output.column, d.output.kind) == ("boxes", "detections")
        assert get_inference_output_kind("object_detection") == "detections"
        assert get_inference_input_spec("object_detection")["kind"] == "file"

    def test_class_annotation_type_says_which_rows_carry_the_class(self):
        from omnitrain.domain.task_registry import class_annotation_type

        assert class_annotation_type("image_classification") == "classification"
        assert class_annotation_type("object_detection") == "bounding_box"
        assert class_annotation_type("token_classification") is None  # classes live in tag data
        assert class_annotation_type("tabular_regression") is None  # no classes at all
        assert class_annotation_type("image_captioning") is None
        assert class_annotation_type(None) is None


class TestTaskPlugins:
    """A task is a class: defining one registers it, with nothing else to edit."""

    @pytest.fixture
    def tasks(self):
        from omnitrain.tasks import TASKS

        TASKS.ensure_loaded()
        before = set(TASKS._items)
        yield TASKS
        for plugin_id in set(TASKS._items) - before:
            TASKS.unregister(plugin_id)

    def test_a_new_class_shows_up_everywhere_the_registry_is_read(self, tasks):
        from omnitrain.tasks import AnnotationSpec, ColumnSpec, ItemSpec, Task, TaskOutput

        class SentimentScore(Task):
            id = "sentiment_score"
            label = "Sentiment Score"
            modality = "text"
            item_spec = ItemSpec(payload="inline_text")
            annotation = AnnotationSpec(type="classification", requires_label_classes=False)
            columns = [ColumnSpec(name="text", kind="inline_text"), ColumnSpec(name="score", kind="label")]
            output = TaskOutput(column="score", kind="regression")
            input_fields = ["text"]

        assert find_task_descriptor("sentiment_score") is not None
        assert list(TASK_REGISTRY)[-1] == "sentiment_score"  # a plugin that sets no `order` sorts last
        assert task_to_modality("sentiment_score") == "text"
        assert get_inference_output_kind("sentiment_score") == "regression"
        assert "sentiment_score" in registry_json()["tasks"]
        assert SentimentScore in tasks.all()

    def test_a_helper_base_without_an_id_is_not_a_task(self, tasks):
        from omnitrain.tasks import Task

        before = len(TASK_REGISTRY)

        class Helper(Task):
            label = "not a task"

        assert len(TASK_REGISTRY) == before

    def test_duplicate_ids_are_refused(self, tasks):
        from omnitrain.tasks import Task

        with pytest.raises(ValueError, match="Duplicate task id"):

            class Again(Task):
                id = "image_classification"
                label = "Imposter"
                modality = "vision"
                item_spec = TASK_REGISTRY["image_classification"].item_spec
                annotation = TASK_REGISTRY["image_classification"].annotation
                columns = TASK_REGISTRY["image_classification"].columns
                output = TASK_REGISTRY["image_classification"].output

    def test_an_incomplete_task_is_refused_at_class_creation(self, tasks):
        from omnitrain.tasks import Task

        with pytest.raises(TypeError, match="is missing"):

            class Nameless(Task):
                id = "nameless"

        with pytest.raises(TypeError, match="needs snapshot columns and an output"):

            class NoOutput(Task):
                id = "no_output"
                label = "No output"
                modality = "text"
                status = "stable"
                item_spec = TASK_REGISTRY["text_classification"].item_spec
                annotation = TASK_REGISTRY["text_classification"].annotation

        assert find_task_descriptor("nameless") is None and find_task_descriptor("no_output") is None
