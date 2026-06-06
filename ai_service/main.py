"""
Theseus AI Microservice -- FastAPI entry point.

Endpoints:
  GET  /health                       - Health check
  GET  /api/v1/models                - List registered models
  POST /api/v1/jobs/train            - Start a training task
  GET  /api/v1/jobs/{task_id}/status - Get task status
  DELETE /api/v1/jobs/{task_id}      - Stop a running training task
  POST /api/v1/jobs/inference        - Run inference on a single image
  POST /api/v1/jobs/export           - Export trained weights
"""

from celery.contrib.abortable import AbortableAsyncResult
import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ai_service.dirs import DATA_DIR, MODEL_DIR, UPLOAD_DIR
from ai_service.dto import ExportRequest, JobResponse, TrainRequest
from ai_service.models import get_model, list_models
from ai_service.worker import celery_app, export_model_task, inference_task, train_model_task

app = FastAPI(
    title="CTU Theseus AI Service",
    description="API gateway for asynchronous PyTorch model training, inference, and exporting.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@app.get("/health")
def health():
    """Health check endpoint for Docker / load-balancer probes."""
    return {"status": "ok"}


@app.get("/api/v1/models")
def get_models(query: Optional[str] = None, family: Optional[str] = None):
    """Returns registered model architectures, optionally filtered by query string or family."""
    models = list_models()

    if family:
        models = [m for m in models if m["family"].lower() == family.lower()]

    if query:
        for model_group in models:
            model_group["variants"] = [
                v for v in model_group["variants"]
                if query.lower() in v["id"].lower() or query.lower() in v["display_name"].lower()
            ]
        models = [m for m in models if m["variants"]]

    return models


@app.post("/api/v1/jobs/train", response_model=JobResponse)
def start_training_job(req: TrainRequest):
    """Validates the configuration against the model registry and queues a training job."""

    # Validate against the model registry
    try:
        get_model(req.model.architecture)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Send to Celery worker queue
    task = train_model_task.delay(req.id, req.model_dump())

    return JobResponse(job_id=task.id, status="queued")


@app.delete("/api/v1/jobs/{task_id}")
def stop_training(task_id: str):
    """Terminate a running training task."""
    AbortableAsyncResult(task_id).abort()
    return {"job_id": task_id, "status": "stopped"}


@app.get("/api/v1/jobs/{task_id}/status")
def get_status(task_id: str):
    """Get the current status of a Celery task."""
    result = celery_app.AsyncResult(task_id)
    response = {"job_id": task_id, "status": result.state}
    if result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.result)
    elif result.info:
        response["progress"] = result.info
    return response


@app.post("/api/v1/jobs/inference", response_model=JobResponse)
def inference(
    model_id: str = Form(...),
    model_name: str = Form(...),
    threshold: float = Form(0.5),
    webhook_url: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """Accepts an image upload, saves it to disk, and queues an inference job."""

    # Validate model weights exist on disk
    weight_path = os.path.join(MODEL_DIR, f"{model_id}.pt")
    if not os.path.exists(weight_path):
        raise HTTPException(status_code=404, detail=f"Model weights for '{model_id}' not found.")

    # Validate model_name is in the registry
    try:
        get_model(model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate filename is present
    if not image.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    # Validate file size
    image.file.seek(0, 2)
    file_size = image.file.tell()
    image.file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    file_ext = image.filename.rsplit(".", 1)[-1]
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{file_ext}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    task = inference_task.delay(model_id, model_name, str(image_path), threshold, webhook_url)

    return JobResponse(job_id=task.id, status="queued", message="Inference job queued")


@app.post("/api/v1/jobs/export", response_model=JobResponse)
def start_export_job(request: ExportRequest):
    """Queues a job to export pure PyTorch weights into ONNX/TorchScript."""

    weight_path = os.path.join(MODEL_DIR, f"{request.model_id}.pt")
    if not os.path.exists(weight_path):
        raise HTTPException(status_code=404, detail=f"Model weights for '{request.model_id}' not found.")

    # Validate model_name is in the registry
    try:
        get_model(request.model_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task = export_model_task.delay(
        request.model_id,
        request.model_name,
        request.num_classes,
        request.export_format,
        request.webhook_url,
    )

    return JobResponse(
        job_id=task.id,
        status="queued",
        message=f"Export job to {request.export_format} queued.",
    )
