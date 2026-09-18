"""OpenTelemetry metric instruments for the worker — the counterpart to the
tracing-only setup in telemetry.py. Exported via the same OTLP pipeline (see
telemetry.py's meter provider), so Aspire's dashboard picks these up
alongside the existing traces.
"""

from opentelemetry import metrics

meter = metrics.get_meter("theseus-worker")

nats_task_duration = meter.create_histogram(
    "theseus.nats_task.duration",
    description="Time spent processing one NATS task/command end to end",
    unit="ms",
)

nats_task_count = meter.create_counter(
    "theseus.nats_task.count",
    description="NATS tasks/commands processed, by subject and outcome",
)
