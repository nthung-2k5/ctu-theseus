import asyncio
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
from ludwig.api import LudwigModel
from nats.aio.msg import Msg
from opentelemetry import trace
from schema.inference_request import InferenceRequest
from services.nats import nats_service
from services.predict import parse_prediction_row
from services.storage import download_model, find_model_dir

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("theseus-worker")


async def _build_input_value(request: InferenceRequest, temp_dir: str) -> str:
    """Resolve the request's payload into the single value Ludwig's
    single-row DataFrame needs for the model's input column — a local file
    path for file-backed modalities, the raw text for text tasks, or the one
    numeric/string value the record payload carries.
    """
    payload = request.payload

    if payload.kind == "file":
        local_path = Path(temp_dir) / payload.upload_filename
        downloaded = await nats_service.download(
            "theseus-inferences", payload.upload_key, local_path
        )
        if not downloaded:
            raise FileNotFoundError(f"Upload not found for key: {payload.upload_key}")
        return str(local_path)

    if payload.kind == "text":
        return payload.text

    raise ValueError(f"Unexpected payload kind for a single-value input: {payload.kind}")


async def handle_inference(msg: Msg) -> None:
    """Handle an inference task message."""
    request = InferenceRequest.model_validate_json(msg.data.decode("utf-8"))
    run_id = str(request.run_id)

    logger.info(f"Starting inference for run {run_id}")

    try:
        # 1. Download the trained model from S3 (cached locally) and locate
        # the actual Ludwig model directory within it.
        model_dir = find_model_dir(download_model(run_id))

        with tempfile.TemporaryDirectory() as temp_dir:
            # 2. Resolve the payload into either a record dict (tabular —
            # one input feature per field) or a single value (every other
            # task has exactly one input feature: a file path or raw text).
            resolved_input: dict[str, str | float] | str
            if request.payload.kind == "record":
                resolved_input = request.payload.record
            else:
                resolved_input = await _build_input_value(request, temp_dir)

            # 3. Load and run inference in a thread
            def _run_inference():
                with tracer.start_as_current_span("inference.predict"):
                    model = LudwigModel.load(model_dir)
                    input_feature = model.config_obj.input_features[0]
                    output_feature = model.config_obj.output_features[0]

                    if isinstance(resolved_input, dict):
                        input_df = pd.DataFrame({col: [val] for col, val in resolved_input.items()})
                    else:
                        input_df = pd.DataFrame({input_feature.column: [resolved_input]})

                    predictions, _ = model.predict(dataset=input_df)
                    assert isinstance(predictions, pd.DataFrame)

                    idx2str = None
                    if (
                        output_feature.type == "category"
                        and model.training_set_metadata
                    ):
                        idx2str = model.training_set_metadata.get(
                            output_feature.name, {}
                        ).get("idx2str")

                    return (
                        output_feature.name,
                        output_feature.type,
                        predictions,
                        idx2str,
                    )

            feature_name, feature_type, predictions, idx2str = await asyncio.to_thread(
                _run_inference
            )

            assert isinstance(predictions, pd.DataFrame)

            # Parse predictions into threshold-filtered results, keyed by
            # class name for classification tasks. Ludwig names prediction
            # columns after the output feature (e.g. "class_probabilities"),
            # not a fixed "label_*" — that name comes from the task registry
            # and varies per task, so it must be read off the loaded model.
            results = parse_prediction_row(
                feature_name, feature_type, predictions, idx2str, request.threshold
            )

            await msg.respond(
                json.dumps(
                    {
                        "status": "success",
                        "results": results,
                    }
                ).encode("utf-8")
            )

            logger.info(f"Inference completed for run {run_id}")

    except Exception as e:
        logger.exception(f"Inference failed for run {run_id}")

        await msg.respond(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(e),
                }
            ).encode("utf-8")
        )

        raise
