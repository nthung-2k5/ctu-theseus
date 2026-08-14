/**
 * NATS JetStream client for the ElysiaJS gateway.
 *
 * Provides connection management, stream provisioning, and publish/subscribe
 * helpers.
 *
 * Topology (mirrored in ai_service/services/nats.py — keep both in sync):
 *
 *   THESEUS_TASKS    theseus.task.train.{runId}, theseus.task.export.{jobId}
 *   THESEUS_EVENTS   theseus.event.run.{runId}.{kind}   (status|metric|log)
 *   THESEUS_COMMANDS theseus.command.run.{runId}
 *
 * Every dispatched-work and event subject carries its id as the trailing
 * token so the worker's `*` wildcard subscriptions match it. Status,
 * metrics and logs all land on the single THESEUS_EVENTS stream — that's
 * what lets an SSE reconnect replay a run's full history from one
 * `Last-Event-ID` (the JetStream stream sequence number).
 */

import {
  AckPolicy,
  type Consumer,
  DeliverPolicy,
  type JetStreamClient,
  type JetStreamManager,
  jetstream,
  jetstreamManager,
  RetentionPolicy,
} from '@nats-io/jetstream'
import { type MsgHdrs, type NatsConnection, headers as natsHeaders } from '@nats-io/nats-core'
import { Objm } from '@nats-io/obj'
import { connect } from '@nats-io/transport-node'
import { context, propagation, trace } from '@opentelemetry/api'
import { config } from '@server/lib/config'

let nc: NatsConnection
let js: JetStreamClient
let jsm: JetStreamManager
let objm: Objm

const NATS_URI = config.natsUri
const tracer = trace.getTracer('theseus-gateway-nats')

/* ------------------------------------------------------------------ */
/*  W3C trace context propagation over NATS message headers           */
/* ------------------------------------------------------------------ */

const headerSetter = { set: (carrier: MsgHdrs, key: string, value: string) => carrier.set(key, value) }
const headerGetter = {
  keys: (carrier: MsgHdrs) => carrier.keys(),
  get: (carrier: MsgHdrs, key: string) => (carrier.has(key) ? carrier.get(key) : undefined),
}

function injectTraceHeaders(): MsgHdrs {
  const h = natsHeaders()
  propagation.inject(context.active(), h, headerSetter)
  return h
}

function extractTraceContext(h: MsgHdrs | undefined) {
  if (!h) return context.active()
  return propagation.extract(context.active(), h, headerGetter)
}

/* ------------------------------------------------------------------ */
/*  Connection & Stream Setup                                         */
/* ------------------------------------------------------------------ */

/**
 * Connect to NATS and provision JetStream streams.
 * Call once at gateway startup.
 */
export async function initNats(): Promise<void> {
  console.log(`[nats] Connecting to ${NATS_URI}...`)

  const parsed = new URL(NATS_URI)
  nc = await connect({
    servers: `${parsed.hostname}:${parsed.port}`,
    user: parsed.username || undefined,
    pass: parsed.password || undefined,
  })
  js = jetstream(nc)
  jsm = await jetstreamManager(nc)
  objm = new Objm(js)

  // Provision streams (idempotent — creates if missing, updates if exists)
  const streams = [
    {
      name: 'THESEUS_TASKS',
      subjects: ['theseus.task.>'],
      retention: RetentionPolicy.Workqueue,
      max_age: 24 * 3600 * 1_000_000_000, // 24h in nanoseconds
    },
    {
      name: 'THESEUS_EVENTS',
      subjects: ['theseus.event.>'],
      retention: RetentionPolicy.Limits,
      max_age: 7 * 24 * 3600 * 1_000_000_000, // 7 days
    },
    {
      name: 'THESEUS_COMMANDS',
      subjects: ['theseus.command.>'],
      retention: RetentionPolicy.Workqueue,
      max_age: 3600 * 1_000_000_000, // 1 hour
    },
  ]

  for (const stream of streams) {
    try {
      await jsm.streams.info(stream.name)
      await jsm.streams.update(stream.name, stream)
      console.log(`[nats] Stream '${stream.name}' updated.`)
    } catch {
      await jsm.streams.add(stream)
      console.log(`[nats] Stream '${stream.name}' created.`)
    }
  }

  // Provision Object Store for inference uploads
  try {
    await objm.create('theseus-inferences', { description: 'Uploads for inference tasks' })
    console.log(`[nats] Object Store 'theseus-inferences' created/verified.`)
  } catch (err) {
    console.error(`[nats] Failed to create Object Store:`, err)
  }

  console.log('[nats] Connected and streams provisioned.')
}

/**
 * Gracefully close the NATS connection.
 */
export async function closeNats(): Promise<void> {
  if (nc) {
    await nc.drain()
    console.log('[nats] Connection closed.')
  }
}

/* ------------------------------------------------------------------ */
/*  Publishing                                                        */
/* ------------------------------------------------------------------ */

/**
 * Publish a JSON message to a JetStream subject. Injects the current span's
 * W3C traceparent into message headers so a consumer on the other side of
 * the NATS hop can continue the same trace.
 */
async function publish(subject: string, data: unknown): Promise<void> {
  const ack = await js.publish(subject, JSON.stringify(data), { headers: injectTraceHeaders() })
  console.log(`[nats] Published to ${subject} (stream=${ack.stream}, seq=${ack.seq})`)
}

/**
 * Publish a training task. Subject carries the run id as the trailing
 * token so the worker's `theseus.task.train.*` subscription matches it.
 */
export async function publishTrainTask(runId: string, data: unknown): Promise<void> {
  await publish(`theseus.task.train.${runId}`, data)
}

/**
 * Publish an export task. Subject carries the export job id (a run can
 * have more than one export, in different formats).
 */
export async function publishExportTask(jobId: string, data: unknown): Promise<void> {
  await publish(`theseus.task.export.${jobId}`, data)
}

/**
 * Publish an abort command for a training run.
 */
export async function publishAbortCommand(runId: string): Promise<void> {
  await publish(`theseus.command.run.${runId}`, {
    command: 'abort',
    runId,
  })
}

/**
 * Publish an inference request/response over core NATS (synchronous,
 * request/reply — not JetStream). Mirrors InferenceRequestSchema in
 * lib/schema.ts — keep both in sync.
 */
export type InferenceInputPayload =
  | { kind: 'file'; uploadKey: string; uploadFilename: string }
  | { kind: 'text'; text: string }
  | { kind: 'record'; record: Record<string, string | number> }

export interface InferencePayload {
  runId: string
  threshold: number
  payload: InferenceInputPayload
}

export type InferenceResponse =
  | {
      status: 'success'
      results: Record<string, number>
    }
  | {
      status: 'failed'
      error: string
    }

export async function requestInferenceTask(runId: string, data: InferencePayload): Promise<InferenceResponse> {
  const msg = await nc.request(`theseus.inference.${runId}`, JSON.stringify(data))
  return await msg.json()
}

/* ------------------------------------------------------------------ */
/*  Object Store                                                      */
/* ------------------------------------------------------------------ */

/**
 * Upload an image to NATS Object Store for inference.
 */
export async function uploadInferenceImage(filename: string, data: Uint8Array): Promise<string> {
  const store = await objm.open('theseus-inferences')
  const info = await store.putBlob({ name: filename }, data)
  return info.name
}

/* ------------------------------------------------------------------ */
/*  Subscribing (pull-based durable consumers)                        */
/* ------------------------------------------------------------------ */

export interface NatsMessage {
  subject: string
  data: unknown
}

/**
 * Subscribe to a JetStream subject with a durable consumer.
 * The handler is called for each message. Messages are auto-acked after
 * successful handler execution.
 *
 * Pass `signal` (e.g. from an AbortController wired to SIGTERM) to stop the
 * background fetch loop cleanly on shutdown.
 */
export async function subscribe<T>(
  stream: string,
  subject: string,
  durableName: string,
  handler: (data: T, subject: string) => Promise<void>,
  signal?: AbortSignal,
): Promise<void> {
  // Ensure consumer exists
  try {
    await jsm.consumers.info(stream, durableName)
  } catch {
    await jsm.consumers.add(stream, {
      durable_name: durableName,
      filter_subject: subject,
      deliver_policy: DeliverPolicy.All,
      ack_policy: AckPolicy.Explicit,
      ack_wait: 30 * 1_000_000_000, // 30 seconds
    })
  }

  const consumer = await js.consumers.get(stream, durableName)

  // Process messages in background
  ;(async () => {
    while (!signal?.aborted) {
      try {
        const messages = await consumer.fetch({ max_messages: 1, expires: 5_000 })
        for await (const msg of messages) {
          try {
            const data = msg.json<T>()
            const parentCtx = extractTraceContext(msg.headers)
            await context.with(parentCtx, () =>
              tracer.startActiveSpan(`nats.consume ${msg.subject}`, async (span) => {
                try {
                  await handler(data, msg.subject)
                } finally {
                  span.end()
                }
              }),
            )
            msg.ack()
          } catch (err) {
            console.error(`[nats] Handler error for ${msg.subject}:`, err)
            msg.nak(10_000) // Retry after 10s
          }
        }
      } catch {
        // Timeout or connection issue — just retry
        if (nc.isClosed()) {
          console.error('[nats] Connection closed, stopping consumer.')
          break
        }
      }
    }
    console.log(`[nats] Consumer '${durableName}' stopped.`)
  })()

  console.log(`[nats] Subscribed: stream=${stream}, subject=${subject}, consumer=${durableName}`)
}

/* ------------------------------------------------------------------ */
/*  Ephemeral consumers (one per SSE connection)                      */
/* ------------------------------------------------------------------ */

/**
 * Create a throwaway consumer on THESEUS_EVENTS scoped to one run, for the
 * SSE route. No durable name — it's deleted when the connection closes, and
 * `inactive_threshold` cleans it up as a safety net if that doesn't happen.
 *
 * `afterSeq` (from the client's `Last-Event-ID`) resumes exactly where a
 * reconnect left off; omit it to replay the run's full history.
 */
export async function createRunEventsConsumer(runId: string, afterSeq?: number): Promise<Consumer> {
  const info = await jsm.consumers.add('THESEUS_EVENTS', {
    filter_subject: `theseus.event.run.${runId}.>`,
    ack_policy: AckPolicy.Explicit,
    ...(afterSeq
      ? { deliver_policy: DeliverPolicy.StartSequence, opt_start_seq: afterSeq + 1 }
      : { deliver_policy: DeliverPolicy.All }),
    inactive_threshold: 5 * 60 * 1_000_000_000, // 5 minutes
  })
  return js.consumers.get('THESEUS_EVENTS', info.name)
}
