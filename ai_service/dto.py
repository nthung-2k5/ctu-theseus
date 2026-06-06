from typing import Optional
from pydantic import BaseModel, Field
from pydantic.types import PositiveInt, NonNegativeFloat
from typing import Literal

class DatasetConfig(BaseModel):
    source_uri: str = Field(..., description="Path or URI to the dataset (e.g., local folder or S3)")
    num_classes: PositiveInt = Field(..., ge=2, description="Number of classes in the dataset")
    batch_size: PositiveInt = Field(64, ge=1, description="Number of images per batch")
    num_workers: PositiveInt = Field(4, ge=0, description="Number of DataLoader workers")

class ModelConfig(BaseModel):
    architecture: str = Field(..., description="Model variant ID from the registry (e.g., resnet50)")
    pretrained: bool = Field(True, description="Whether to start with pre-trained weights")
    drop_rate: NonNegativeFloat = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")

class OptimizationConfig(BaseModel):
    optimizer: Literal["adamw", "adam", "sgd"] = Field("adamw", description="Optimizer type")
    learning_rate: NonNegativeFloat = Field(0.001, gt=0.0, description="Initial learning rate")
    weight_decay: NonNegativeFloat = Field(0.05, ge=0.0, description="Weight decay for regularization")

class ScheduleConfig(BaseModel):
    epochs: PositiveInt = Field(50, ge=1, description="Total number of training epochs")

class TrainRequest(BaseModel):
    """Comprehensive payload for submitting a new training job."""
    id: str = Field(..., description="ID of the training run")
    dataset: DatasetConfig
    model: ModelConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    webhook_url: Optional[str] = Field(None, description="Optional webhook URL to receive progress and completion events")

class JobResponse(BaseModel):
    """Standard response for job operations."""
    job_id: str
    status: str
    message: str | None = None

class ExportRequest(BaseModel):
    """Payload for exporting a trained model."""
    model_id: str = Field(..., description="The unique ID of the trained model")
    model_name: str = Field(..., description="The architectural name (e.g., resnet18)")
    num_classes: int = Field(..., description="Number of output classes the model was trained on")
    export_format: str = Field("onnx", pattern="^(onnx|torchscript)$", description="Target format for export")
    webhook_url: Optional[str] = Field(None, description="Optional webhook URL to receive completion events")
