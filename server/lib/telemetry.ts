/**
 * OpenTelemetry setup for the gateway.
 */

import { opentelemetry, record } from '@elysia/opentelemetry'
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics'
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-node'

export const telemetry = opentelemetry({
  serviceName: process.env.OTEL_SERVICE_NAME ?? 'ctu-theseus-gateway',
  spanProcessors: [new BatchSpanProcessor(new OTLPTraceExporter())],
  // Same OTLP pipeline as traces — Aspire's dashboard ingests OTLP metrics
  // directly, so this needs no separate /metrics scrape endpoint. See
  // server/lib/metrics.ts for the actual instruments.
  metricReader: new PeriodicExportingMetricReader({ exporter: new OTLPMetricExporter() }),
})

export { record }
