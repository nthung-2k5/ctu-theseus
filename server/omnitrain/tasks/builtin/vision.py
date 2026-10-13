from omnitrain.tasks.base import AnnotationSpec, ItemSpec, Task, TaskOutput
from omnitrain.tasks.builtin._common import (
    BOXES,
    IMAGE_MIME,
    IMAGE_PATH,
    SPLIT,
    ClassificationTask,
    FileCaptionTask,
    PlannedTask,
    col,
)


class ImageClassification(ClassificationTask):
    id = "image_classification"
    label = "Image Classification"
    modality = "vision"
    status = "stable"
    order = 10
    item_spec = ItemSpec(payload="file", accept=IMAGE_MIME)
    source_column = col(IMAGE_PATH, "storage_uri")


class ImageCaptioning(FileCaptionTask):
    id = "image_captioning"
    label = "Image Captioning"
    modality = "vision"
    order = 110
    path_column = IMAGE_PATH
    output_field = "caption"
    accept = IMAGE_MIME


class ObjectDetection(Task):
    """`boxes` holds a JSON list of {label, x, y, w, h} per image (normalised, top-left origin; see
    domain/boxes.py)."""

    id = "object_detection"
    label = "Object Detection"
    modality = "vision"
    order = 150
    item_spec = ItemSpec(payload="file", accept=IMAGE_MIME)
    annotation = AnnotationSpec(type="bounding_box", requires_label_classes=True)
    columns = [col(IMAGE_PATH, "storage_uri"), col(BOXES, "boxes"), col(SPLIT, "split")]
    output = TaskOutput(column=BOXES, kind="detections")


class ImageSegmentation(PlannedTask):
    id = "image_segmentation"
    label = "Image Segmentation"
    modality = "vision"
    order = 160
    annotation = AnnotationSpec(type="segmentation_mask", requires_label_classes=True)
