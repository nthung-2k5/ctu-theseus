/**
 * OpenTelemetry metric instruments for the gateway — the counterpart to the
 * tracing-only setup in telemetry.ts. Exported via the same OTLP pipeline
 * (see telemetry.ts's `metricReader`), so Aspire's dashboard picks these up
 * automatically alongside the existing traces.
 */

import { metrics } from '@opentelemetry/api'

const meter = metrics.getMeter('ctu-theseus-gateway')

/** How long a single NATS event handler invocation took, by consumer and event kind. */
export const natsEventDuration = meter.createHistogram('theseus.nats_event.duration', {
  description: 'Time spent processing one NATS event end to end',
  unit: 'ms',
})

/** NATS events processed, by consumer, event kind, and outcome */
export const natsEventCount = meter.createCounter('theseus.nats_event.count', {
  description: 'NATS events processed, by consumer/kind/outcome',
})

/** Training runs reaching a terminal status, by that status — the success/failure/cancel rate. */
export const trainingRunTerminalCount = meter.createCounter('theseus.training_run.terminal_count', {
  description: 'Training runs reaching a terminal status (succeeded/failed/canceled)',
})

/**
 * Wraps a NATS event handler with duration + outcome instrumentation. Never
 * changes the handler's behavior or return value — only observes it — so
 * wrapping an existing `subscribe()` call is safe to do at the call site
 * without touching the handler body itself.
 */
export function instrumentNatsHandler<T>(
  consumer: string,
  labelFor: (data: T) => string,
  handler: (data: T, subject: string) => Promise<void>,
): (data: T, subject: string) => Promise<void> {
  return async (data, subject) => {
    const kind = labelFor(data)
    const start = performance.now()
    try {
      await handler(data, subject)
      natsEventCount.add(1, { consumer, kind, outcome: 'success' })
    } catch (error) {
      natsEventCount.add(1, { consumer, kind, outcome: 'error' })
      throw error
    } finally {
      natsEventDuration.record(performance.now() - start, { consumer, kind })
    }
  }
}
