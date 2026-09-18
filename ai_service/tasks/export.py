import asyncio
import logging
import os
import tempfile
from typing import Any

import pandas as pd
from constants import BUCKET_DATASETS, SPLIT_COLUMN_NAME
from ludwig.api import LudwigModel
from schema.export_task import ExportTask
from services.nats import nats_service
from services.predict import parse_prediction_row
from services.storage import (
    BUCKET_EXPORTS,
    cleanup_temp,
    download_model,
    find_model_dir,
    s3fs_readable_path,
    upload_file,
    upload_json,
)

logger = logging.getLogger(__name__)


def _plain(v: Any) -> Any:
    """Converts a pandas/numpy scalar to a plain Python one. json.dumps
    (services/storage.py's upload_json) has no numpy support and would
    otherwise silently stringify it via its `default=str` fallback instead
    of writing a real number."""
    return v.item() if hasattr(v, "item") else v


def _shape_input_value(input_features: list[Any], sample_row: "pd.Series") -> tuple[str | None, Any]:
    """Returns (inputColumn, inputValue) for expected.json.

    A single input feature keeps the original scalar shape — an `s3://` URI
    for file-backed modalities, or the inline text/scalar value — which
    bundle.ts's resolveSampleFile turns into sample/input.<ext>. A tabular
    model (more than one input feature) becomes a {column: value} record
    instead, written as sample/input.json; theseus_client.py's predict()
    takes that same dict shape for a tabular model.
    """
    if len(input_features) == 1:
        return input_features[0].column, _plain(sample_row[input_features[0].column])
    return None, {f.column: _plain(sample_row[f.column]) for f in input_features}


def _build_golden_sample(ludwig_model_dir: str, dataset_key: str) -> dict[str, Any] | None:
    """Run one real test-split row through the trained model and return the
    input/output pair the devkit's `verify` script checks its own
    (re-implemented) preprocessing against. Returns None rather than raising
    — a missing/unreadable snapshot shouldn't fail the export itself, just
    skip the verification artifact.
    """
    try:
        model = LudwigModel.load(ludwig_model_dir)
        df = pd.read_parquet(s3fs_readable_path(BUCKET_DATASETS, dataset_key))
        rows = df[df[SPLIT_COLUMN_NAME] == "test"] if SPLIT_COLUMN_NAME in df.columns else df
        if len(rows) == 0:
            rows = df
        sample = rows.iloc[[0]]

        input_features = model.config_obj.input_features
        output_feature = model.config_obj.output_features[0]

        predictions, _ = model.predict(dataset=sample)
        assert isinstance(predictions, pd.DataFrame)

        idx2str = None
        if output_feature.type == "category" and model.training_set_metadata:
            idx2str = model.training_set_metadata.get(output_feature.name, {}).get("idx2str")

        # threshold=0.0: this is a reference fixture, not a filtered result —
        # the devkit's verify script wants the full distribution to compare.
        predicted = parse_prediction_row(
            output_feature.name, output_feature.type, predictions, idx2str, threshold=0.0
        )

        input_column, input_value = _shape_input_value(input_features, sample.iloc[0])

        return {
            "schemaVersion": 1,
            "inputColumn": input_column,
            "inputValue": input_value,
            "outputColumn": output_feature.name,
            "outputType": output_feature.type,
            "predictions": predicted,
        }
    except (AttributeError, TypeError, KeyError, IndexError, NameError):
        # A bug in this function, not a data problem — surface it. These are
        # exactly the failures that previously vanished into `return None`,
        # silently disabling every bundle's verify script.
        logger.exception("Bug while building the golden sample for export verification")
        raise
    except Exception:
        # Genuinely environmental (unreadable snapshot, S3 hiccup, a model that
        # won't load): skip the verification artifact rather than failing the
        # export itself.
        logger.exception("Failed to build golden sample for export verification — skipping expected.json")
        return None


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

        # 3. Verification fixture for the devkit bundle (best-effort — see
        # _build_golden_sample). Only makes sense once per run, not once per
        # format, but re-running it is cheap and idempotent (same S3 key).
        if data.dataset_key:
            golden = await asyncio.to_thread(_build_golden_sample, ludwig_model_dir, data.dataset_key)
            if golden is not None:
                upload_json(BUCKET_EXPORTS, f"{model_id}/expected.json", golden)

        await nats_service.publish_export_event(
            model_id, job_id, "success", format=export_format, export_key=export_s3_key
        )

        logger.info(f"Export completed for job {job_id}")

        # Clean up local temp
        cleanup_temp("exports", job_id)

    except Exception:
        # No terminal "failed" event here — see `on_export_permanent_failure`
        # below, which fires exactly once instead of once per retry attempt.
        logger.exception(f"Export failed for job {job_id}")
        raise


async def on_export_permanent_failure(data: ExportTask, error: str) -> None:
    """Called once an `ExportTask` has exhausted all delivery attempts (see
    `services/nats.py` `_consume_loop`'s `on_permanent_failure`)."""
    logger.error(f"Export permanently failed for job {data.job_id} after retries: {error}")
    await nats_service.publish_export_event(str(data.run_id), str(data.job_id), "failed", error=error)
