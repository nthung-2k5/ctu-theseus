"""OpenTelemetry setup. Only active when an OTLP endpoint is configured (Aspire injects one)."""

import logging
import os

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def init_telemetry(app: FastAPI) -> bool:
    if not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")):
        return False

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "ctu-theseus")})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    # Same OTLP pipeline as traces: the Aspire dashboard ingests OTLP metrics directly.
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    )

    FastAPIInstrumentor.instrument_app(app)
    # The engine is created lazily, so instrument the class-level hook rather than an instance.
    SQLAlchemyInstrumentor().instrument()
    logger.info("OpenTelemetry initialized.")
    return True
