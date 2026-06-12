import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ludwig.api import LudwigModel

from ai_service.services.nats import nats_service
from ai_service.services.storage import (
    BUCKET_EXPORTS,
    cleanup_temp,
    download_model,
    upload_file,
)

logger = logging.getLogger(__name__)


async def handle_export(data: dict[str, Any]) -> None:
    """Handle an export task message."""
    job_id = data["id"]
    model_id = data["model_id"]
    export_format = data.get("export_format", "onnx")

    logger.info(
        f"Starting export job {job_id} for model {model_id} (format: {export_format})"
    )

    try:
        # 1. Download model from S3 (cached locally)
        model_save_dir = download_model(model_id)

        # Find the Ludwig model output directory
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if "model_hyperparameters.json" in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            ludwig_model_dir = model_save_dir

        import tempfile

        export_path = os.path.join(
            tempfile.gettempdir(),
            "theseus",
            "exports",
            job_id,
            f"model.{export_format}",
        )
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        def _run_export():
            model = LudwigModel.load(ludwig_model_dir)
            if export_format == "torchscript":
                model.export_model(export_path, format="torch_export")
            elif export_format == "onnx":
                model.export_model(export_path, format="onnx")
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            return export_path

        result_path = await asyncio.to_thread(_run_export)

        # 2. Upload exported model to S3
        export_s3_key = f"{model_id}/model.{export_format}"
        upload_file(BUCKET_EXPORTS, export_s3_key, result_path)

        await nats_service.publish_result(
            "export",
            job_id,
            {
                "id": job_id,
                "type": "export",
                "status": "success",
                "result": {
                    "format": export_format,
                    "export_key": export_s3_key,
                },
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(f"Export completed for job {job_id}")

        # Clean up local temp
        cleanup_temp("exports", job_id)

    except Exception as e:
        logger.exception(f"Export failed for job {job_id}")
        await nats_service.publish_result(
            "export",
            job_id,
            {
                "id": job_id,
                "type": "export",
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
