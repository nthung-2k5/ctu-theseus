"""OpenTelemetry instruments. Exported through the same OTLP pipeline as traces (see telemetry.py)."""

from opentelemetry import metrics

meter = metrics.get_meter("theseus")

training_run_terminal_count = meter.create_counter(
    "theseus.training_run.terminal_count",
    description="Training runs reaching a terminal status (succeeded/failed/canceled)",
)

job_duration = meter.create_histogram(
    "theseus.job.duration",
    description="Time spent running one job end to end, by lane and outcome",
    unit="ms",
)

job_count = meter.create_counter(
    "theseus.job.count",
    description="Jobs finished, by lane and outcome",
)
