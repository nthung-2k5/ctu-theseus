"""Import every model so Base.metadata is complete (Alembic and create_all depend on this)."""

from theseus.db.models.auth import ApiKey, RefreshToken, User
from theseus.db.models.dataset import (
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
    VisionFeatures,
)
from theseus.db.models.serving import ModelExport
from theseus.db.models.training import RunEvaluation, RunEvent, Sweep, TrainingMetric, TrainingRun

__all__ = [
    "Annotation", "ApiKey", "AudioFeatures", "Dataset", "DatasetItem", "DatasetVersion",
    "DatasetVersionItem", "LabelClass", "ModelExport", "Project", "RefreshToken",
    "RunEvaluation", "RunEvent", "Sweep", "TabularFeatures", "TextFeatures", "TrainingMetric",
    "TrainingRun", "User", "VisionFeatures",
]  # fmt: skip
