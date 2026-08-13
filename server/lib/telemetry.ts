/**
 * OpenTelemetry setup for the gateway.
 */

import { opentelemetry, record } from '@elysia/opentelemetry'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http'
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-node'

export const telemetry = opentelemetry({
  serviceName: process.env.OTEL_SERVICE_NAME ?? 'ctu-theseus-gateway',
  spanProcessors: [new BatchSpanProcessor(new OTLPTraceExporter())],
})

export { record }
