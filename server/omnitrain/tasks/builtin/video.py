from omnitrain.tasks.base import ItemSpec
from omnitrain.tasks.builtin._common import VIDEO_MIME, VIDEO_PATH, ClassificationTask, col


class VideoClassification(ClassificationTask):
    """One class per clip, like audio classification."""

    id = "video_classification"
    label = "Video Classification"
    modality = "video"
    order = 140
    item_spec = ItemSpec(payload="file", accept=VIDEO_MIME)
    source_column = col(VIDEO_PATH, "storage_uri")
