import asyncio
import logging
import os
import tempfile

from ludwig.api import LudwigModel
from schema.export_task import ExportTask
from services.nats import nats_service
from services.storage import (
    BUCKET_EXPORTS,
    cleanup_temp,
    download_model,
    find_model_dir,
    upload_file,
)

logger = logging.getLogger(__name__)


async def handle_export(data: ExportTask) -> None:
    """Handle an export task message."""
    job_id = str(data.job_id)
    # The exported model is the trained run's output; run_id doubles as the
    # model identifier until the S3 layout rework gives exports their own key.
    model_id = str(data.run_id)
    export_format = data.format

    logger.info(
        f"Starting export job {job_id} for model {model_id} (format: {export_format})"
    )

    try:
        # 1. Download model from S3 (cached locally)
        ludwig_model_dir = find_model_dir(download_model(model_id))

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

        # NOTE: not part of the formal status|metric|log RunEvent union yet —
        # nothing consumes export completion events on the gateway side
        # until export dispatch is properly wired up.
        await nats_service.publish_event(
            model_id,
            "export",
            {
                "jobId": job_id,
                "status": "success",
                "format": export_format,
                "exportKey": export_s3_key,
            },
        )

        logger.info(f"Export completed for job {job_id}")

        # Clean up local temp
        cleanup_temp("exports", job_id)

    except Exception as e:
        logger.exception(f"Export failed for job {job_id}")
        await nats_service.publish_event(
            model_id,
            "export",
            {"jobId": job_id, "status": "failed", "error": str(e)},
        )
        raise
