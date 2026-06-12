import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from ludwig.api import LudwigModel
from ludwig.utils.types import DataFrame
from nats.aio.msg import Msg
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from ai_service.services.nats import nats_service
from ai_service.services.storage import download_model

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)

    upload_key: str
    upload_filename: str
    threshold: float = 0.5


async def handle_inference(msg: Msg) -> None:
    """Handle an inference task message."""
    request = InferenceRequest.model_validate_json(msg.data.decode("utf-8"))
    model_id = msg.subject.split(".")[-1]

    logger.info(f"Starting inference job for model {model_id}")

    try:
        # 1. Download model from S3 (cached locally)
        model_save_dir = download_model(model_id)
        map_path = os.path.join(model_save_dir, "class_mapping.json")

        with open(map_path, "r") as f:
            class_mapping = json.load(f)

        # Find the Ludwig model output directory
        ludwig_model_dir = None
        for root, dirs, files in os.walk(model_save_dir):
            if "model_hyperparameters.json" in files:
                ludwig_model_dir = root
                break

        if ludwig_model_dir is None:
            ludwig_model_dir = model_save_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            local_image_dir = Path(temp_dir)
            local_image_path = local_image_dir / request.upload_filename

            # 2. Download inference input image from NATS Object Store
            downloaded = await nats_service.download(
                "theseus-inferences", request.upload_key, local_image_path
            )
            if not downloaded:
                raise FileNotFoundError(
                    f"Upload not found for key: {request.upload_key}"
                )

            # 3. Load and run inference in a thread
            def _run_inference():
                model = LudwigModel.load(ludwig_model_dir)
                input_df = pd.DataFrame({"image_path": [local_image_path]})
                predictions, _ = model.predict(dataset=input_df)
                return predictions

            predictions = await asyncio.to_thread(_run_inference)

            assert isinstance(predictions, DataFrame)

            # Parse predictions into threshold-filtered results
            results = {}
            if "label_probabilities" in predictions.columns:
                probs = predictions["label_probabilities"].iloc[0]
                if isinstance(probs, list):
                    idx_to_class = {v: k for k, v in class_mapping.items()}
                    for idx, prob in enumerate(probs):
                        if prob >= request.threshold and idx in idx_to_class:
                            results[idx_to_class[idx]] = round(float(prob), 4)
                elif isinstance(probs, dict):
                    for class_name, prob in probs.items():
                        if prob >= request.threshold:
                            results[class_name] = round(float(prob), 4)

            if not results and "label_predictions" in predictions.columns:
                predicted_label = predictions["label_predictions"].iloc[0]
                results[str(predicted_label)] = 1.0

            # Sort results by confidence descending
            results = dict(
                sorted(results.items(), key=lambda item: item[1], reverse=True)
            )

            # Publish result
            await msg.respond(
                json.dumps(
                    {
                        "status": "success",
                        "results": results,
                    }
                ).encode("utf-8")
            )

            logger.info(f"Inference completed for job {model_id}")

    except Exception as e:
        logger.exception(f"Inference failed for job {model_id}")

        await msg.respond(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(e),
                }
            ).encode("utf-8")
        )

        raise
