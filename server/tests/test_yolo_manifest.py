import pytest

from theseus.backends.yolo import YoloBackend
from theseus.backends.yolo.manifest import METADATA_FILENAME, build_preprocessing_manifest

META = {
    "schemaVersion": 1,
    "task": "classify",
    "imgsz": 320,
    "imageColumn": "image_path",
    "outputName": "class",
    "names": ["dog", "cat", "bird"],
}


def test_classes_come_from_the_trained_models_own_order():
    manifest = build_preprocessing_manifest("run-1", {"classNames": ["bird", "cat", "dog"]}, META)

    (output,) = manifest["outputs"]
    # The compiled config lists the classes alphabetically; the model learned a different order. The model wins.
    assert output["classes"] == ["dog", "cat", "bird"]
    assert (output["name"], output["column"], output["type"]) == ("class", "class", "category")
    assert (manifest["schemaVersion"], manifest["runId"]) == (1, "run-1")


def test_input_describes_how_to_prepare_an_image_for_this_model():
    (entry,) = build_preprocessing_manifest("r", {}, META)["inputs"]

    assert (entry["name"], entry["type"], entry["column"]) == ("images", "image", "image_path")
    # Crop, not squash: the generated clients only do that when the manifest asks for it.
    assert entry["imageResize"]["mode"] == "shorter_edge_center_crop"
    assert entry["imageNormalization"] == {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
    # The key the generated clients take the target size from; without it they would fall back to 224.
    assert entry["ludwigPreprocessing"] == {"height": 320, "width": 320}


def test_missing_metadata_is_an_error():
    with pytest.raises(ValueError, match="names.json"):
        build_preprocessing_manifest("r", {}, None)


def test_the_backend_exposes_the_metadata_file_name_the_export_looks_for():
    assert YoloBackend.metadata_filename == METADATA_FILENAME == "names.json"
    assert YoloBackend.preprocessing_manifest("r", {}, META)["outputs"][0]["classes"] == META["names"]
