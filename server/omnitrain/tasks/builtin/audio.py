from omnitrain.tasks.base import AnnotationSpec, ItemSpec
from omnitrain.tasks.builtin._common import (
    AUDIO_MIME,
    AUDIO_PATH,
    ClassificationTask,
    FileCaptionTask,
    PlannedTask,
    col,
)


class AudioClassification(ClassificationTask):
    id = "audio_classification"
    label = "Audio Classification"
    modality = "audio"
    status = "stable"
    order = 50
    item_spec = ItemSpec(payload="file", accept=AUDIO_MIME)
    source_column = col(AUDIO_PATH, "storage_uri")


class AudioCaptioning(FileCaptionTask):
    id = "audio_captioning"
    label = "Audio Captioning"
    modality = "audio"
    order = 120
    path_column = AUDIO_PATH
    output_field = "caption"
    accept = AUDIO_MIME


class AutomaticSpeechRecognition(FileCaptionTask):
    id = "automatic_speech_recognition"
    label = "Automatic Speech Recognition"
    modality = "audio"
    order = 130
    path_column = AUDIO_PATH
    output_field = "transcript"
    accept = AUDIO_MIME


class AudioSegmentation(PlannedTask):
    id = "audio_segmentation"
    label = "Audio Segmentation"
    modality = "audio"
    order = 170
    annotation = AnnotationSpec(type="segmentation_mask", requires_label_classes=True)
