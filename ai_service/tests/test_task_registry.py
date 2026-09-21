import json
from typing import get_args

import pytest

from theseus.db.enums import ProjectTask
from theseus.services.task_registry import (
    TASK_REGISTRY,
    get_inference_input_spec,
    get_inference_output_kind,
    get_task_descriptor,
    is_classification_task,
    list_selectable_tasks,
    registry_json,
    task_to_modality,
)


def test_every_project_task_has_a_descriptor_and_no_extras():
    assert set(TASK_REGISTRY) == set(get_args(ProjectTask))


def test_thirteen_ludwig_tasks_and_six_planned():
    assert len(list_selectable_tasks()) == 13
    planned = [t for t in TASK_REGISTRY.values() if t.backend == "unsupported"]
    assert len(planned) == 6
    assert all(t.status == "planned" and t.ludwig is None and t.columns == [] for t in planned)


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
        # planned tasks default to a file payload, but have no Ludwig config to derive text fields from
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
            get_inference_output_kind("object_detection")


def test_descriptors_are_immutable_data_the_compiler_can_not_corrupt():
    # Compiling deep-copies feature dicts, so the shared registry must be unchanged afterwards.
    before = get_task_descriptor("image_classification").model_dump()
    from theseus.services.ludwig_config import TrainerSelections, compile_ludwig_config
    from theseus.services.task_registry import SnapshotContext

    compile_ludwig_config(
        get_task_descriptor("image_classification"),
        SnapshotContext(),
        TrainerSelections(image_size=64, augmentations=["random_rotate"]),
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
    assert qa["ludwig"]["trainerKnobs"]["batchSize"]["default"] == 1
    assert "inferenceInputSpec" not in data["tasks"]["object_detection"]
    vision = data["tasks"]["image_classification"]["ludwig"]["encoders"][0]
    assert vision == {
        "id": "resnet18",
        "label": "ResNet-18",
        "encoderType": "resnet",
        "pretrained": True,
        "params": {"model_variant": 18},
    }
