"""Import every model so Base.metadata is complete (Alembic and create_all depend on this)."""

from omnitrain.db.models.admin import PluginSetting
from omnitrain.db.models.auth import ApiKey, RefreshToken, User
from omnitrain.db.models.custom_model import CustomModel
from omnitrain.db.models.dataset import (
    Annotation,
    AudioFeatures,
    Dataset,
    DatasetItem,
    DatasetVersion,
    DatasetVersionItem,
    LabelClass,
    Project,
    TabularFeatures,
    TextFeatures,
    VideoFeatures,
    VisionFeatures,
)
from omnitrain.db.models.integration import UserIntegration
from omnitrain.db.models.serving import ModelExport
from omnitrain.db.models.training import RunEvaluation, RunEvent, Sweep, TrainingMetric, TrainingRun

__all__ = [
    "Annotation", "ApiKey", "AudioFeatures", "CustomModel", "Dataset", "DatasetItem", "DatasetVersion",
    "DatasetVersionItem", "LabelClass", "ModelExport", "PluginSetting", "Project", "RefreshToken",
    "RunEvaluation", "RunEvent", "Sweep", "TabularFeatures", "TextFeatures", "TrainingMetric",
    "TrainingRun", "User", "UserIntegration", "VideoFeatures", "VisionFeatures",
]  # fmt: skip
