import asyncio
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
from ludwig.api import LudwigModel
from nats.aio.msg import Msg
from opentelemetry import trace

from ai_service.schema.inference_request import InferenceRequest
from ai_service.services.nats import nats_service
from ai_service.services.storage import download_model, find_model_dir

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("theseus-worker")


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
            local_upload_path = Path(temp_dir) / request.upload_filename

            # 2. Download inference input from NATS Object Store
            downloaded = await nats_service.download(
                "theseus-inferences", request.upload_key, local_upload_path
            )
            if not downloaded:
                raise FileNotFoundError(
                    f"Upload not found for key: {request.upload_key}"
                )

            # 3. Load and run inference in a thread
            def _run_inference():
                with tracer.start_as_current_span("inference.predict"):
                    model = LudwigModel.load(model_dir)
                    input_feature = model.config_obj.input_features[0]
                    output_feature = model.config_obj.output_features[0]

                    input_df = pd.DataFrame(
                        {input_feature.column: [str(local_upload_path)]}
                    )
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
            prob_col = f"{feature_name}_probabilities"
            pred_col = f"{feature_name}_predictions"

            results: dict[str, float] = {}
            if feature_type == "category" and prob_col in predictions.columns:
                probs = predictions[prob_col].iloc[0]
                if idx2str and isinstance(probs, (list, tuple)):
                    for idx, prob in enumerate(probs):
                        if prob >= request.threshold and idx < len(idx2str):
                            results[str(idx2str[idx])] = round(float(prob), 4)
                elif isinstance(probs, dict):
                    for class_name, prob in probs.items():
                        if prob >= request.threshold:
                            results[str(class_name)] = round(float(prob), 4)

            if not results and pred_col in predictions.columns:
                predicted = predictions[pred_col].iloc[0]
                if feature_type == "number":
                    results[feature_name] = round(float(predicted), 4)
                else:
                    results[str(predicted)] = 1.0

            # Sort results by confidence descending
            results = dict(
                sorted(results.items(), key=lambda item: item[1], reverse=True)
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
