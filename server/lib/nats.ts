/**
 * NATS JetStream client for the ElysiaJS gateway.
 *
 * Provides connection management, stream provisioning, and publish/subscribe
 * helpers.
 *
 * Topology (subject/stream definitions live in lib/subjects.ts — the single
 * source of truth, mirrored into ai_service/schema/subjects.py by
 * compile_schema.ts):
 *
 *   THESEUS_TASKS        theseus.task.train.{runId}, theseus.task.export.{jobId}
 *   THESEUS_EVENTS       theseus.event.run.{runId}.{kind}   (status|metric|log|export)
 *   THESEUS_COMMANDS     theseus.command.run.{runId}
 *   THESEUS_DLQ          theseus.dlq.{kind}.{id}            — permanently-failed messages
 *   THESEUS_ABORT_FLAGS  theseus.abortflag.{runId}          — last-value-per-subject
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
import { STREAM_DEFS, type StreamDef, subject } from '@server/lib/subjects'

let nc: NatsConnection
let js: JetStreamClient
let jsm: JetStreamManager
let objm: Objm

const NATS_URI = config.natsUri
const tracer = trace.getTracer('theseus-gateway-nats')

function toStreamConfig(def: StreamDef) {
  return {
    name: def.name,
    subjects: [...def.subjects],
    retention: def.retention === 'workqueue' ? RetentionPolicy.Workqueue : RetentionPolicy.Limits,
    max_age: def.maxAgeSeconds * 1_000_000_000,
    ...(def.maxMsgsPerSubject !== undefined ? { max_msgs_per_subject: def.maxMsgsPerSubject } : {}),
  }
}

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
  const streams = STREAM_DEFS.map(toStreamConfig)

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
  await publish(subject.trainTask(runId), data)
}

/**
 * Publish an export task. Subject carries the export job id (a run can
 * have more than one export, in different formats).
 */
export async function publishExportTask(jobId: string, data: unknown): Promise<void> {
  await publish(subject.exportTask(jobId), data)
}

/**
 * Publish an abort command for a training run, and durably record the
 * abort intent in THESEUS_ABORT_FLAGS (see `setAbortFlag`). The command is
 * the immediate wake-up signal; the flag is what survives a worker restart
 * — `theseus.command.run.*` uses `DeliverPolicy.New` on the worker side, so
 * a worker that's down when this publishes would otherwise never learn
 * about the abort at all.
 */
export async function publishAbortCommand(runId: string): Promise<void> {
  await Promise.all([
    publish(subject.abortCommand(runId), { command: 'abort', runId }),
    setAbortFlag(runId),
  ])
}

/**
 * Dispatch one inference job onto THESEUS_TASKS (JetStream — not the old
 * core-NATS request/reply this replaced). Mirrors InferenceTaskSchema in
 * lib/schema.ts — keep both in sync, or better, change the schema and run
 * `bun run server/compile_schema.ts`.
 */
export type InferenceInputPayload =
  | { kind: 'file'; uploadKey: string; uploadFilename: string }
  | { kind: 'text'; fields: Record<string, string> }
  | { kind: 'record'; record: Record<string, string | number> }
  | { kind: 'batch'; uploadKey: string; uploadFilename: string }

export interface InferenceTaskPayload {
  inferenceId: string
  runId: string
  topK?: number
  payload: InferenceInputPayload
}

export type InferenceOutput =
  | { kind: 'classification'; feature: string; classes: { label: string; confidence: number }[] }
  | { kind: 'regression'; feature: string; value: number }
  | { kind: 'text'; feature: string; text: string }
  | { kind: 'tokens'; feature: string; tokens: { token: string; tag: string }[] }

/**
 * The terminal outcome of one inference job — published once to
 * `theseus.inference.result.{inferenceId}` and persisted to Postgres by
 * `lib/microservice.ts`'s `startInferenceResultsConsumer` the moment it
 * arrives (see `inferenceJobs` in db/schema.ts), rather than read back from
 * NATS on each poll — that stream self-expires after an hour, so a lazy
 * read would lose any result nobody happened to poll for in time.
 */
export type InferenceResponse =
  | {
      status: 'success'
      runId: string
      output: InferenceOutput
    }
  | {
      // Distinct from 'success': a batch job has no single InferenceOutput
      // to inline, only a downloadable results file. resultKey is a
      // BUCKET_MODELS key (see ai_service's batch_inference_result_key).
      status: 'batch'
      runId: string
      resultKey: string
      rowCount: number
    }
  | {
      status: 'failed'
      runId: string
      error: string
    }

/** Publish an inference job for the worker's `inference-worker` consumer to pick up. */
export async function publishInferenceTask(inferenceId: string, data: InferenceTaskPayload): Promise<void> {
  await publish(subject.inferenceTask(inferenceId), data)
}

/**
 * Fire-and-forget signal to preload a run's model into the worker's cache
 * (see ai_service/services/model_cache.py) before the user actually submits
 * an inference request — closes the cold-start gap for the *first* request
 * on a run, which the model cache alone can't help with. Core NATS publish,
 * not a request — the caller doesn't wait for (or need) a reply.
 */
export function publishInferenceWarm(runId: string): void {
  nc.publish(subject.inferenceWarm(runId))
}

/* ------------------------------------------------------------------ */
/*  Abort flags (durable "last value wins" store, no separate KV pkg) */
/* ------------------------------------------------------------------ */

/**
 * Record that a run should be aborted. THESEUS_ABORT_FLAGS is configured
 * with `max_msgs_per_subject: 1`, so this simply overwrites any previous
 * flag for the run — the same last-value-per-subject mechanism NATS's own
 * KV feature is built on, used directly here instead of pulling in a
 * separate KV client package.
 */
export async function setAbortFlag(runId: string): Promise<void> {
  await js.publish(subject.abortFlag(runId), JSON.stringify({ runId, abortedAt: new Date().toISOString() }))
}

/**
 * Publish a permanently-failed message to the dead-letter stream for
 * operator inspection/replay. `kind` distinguishes task/command origin
 * (e.g. 'train', 'export', 'command'); `id` is the run/job id.
 */
export async function publishToDlq(
  kind: string,
  id: string,
  data: { originalSubject: string; payload: unknown; error: string; deliveryCount: number },
): Promise<void> {
  await publish(subject.dlq(kind, id), data)
}

/* ------------------------------------------------------------------ */
/*  Object Store                                                      */
/* ------------------------------------------------------------------ */

/**
 * Upload an inference input file (image, audio, or any other file-backed
 * modality) to the NATS Object Store, keyed by a server-generated name.
 */
export async function uploadInferenceFile(filename: string, data: Uint8Array): Promise<string> {
  const store = await objm.open('theseus-inferences')
  const info = await store.putBlob({ name: filename }, data)
  return info.name
}

/**
 * Delete every `theseus-inferences` object older than `olderThanMs`. The
 * worker deletes an upload itself once its job succeeds or permanently
 * fails (`nats_service.delete_upload` in ai_service/tasks/inference.py) —
 * this is the backstop for uploads whose job never reaches that point
 * (crash, a message that never reaches a worker, a permanently stuck
 * retry loop). Returns the number deleted.
 */
export async function sweepStaleInferenceUploads(olderThanMs: number): Promise<number> {
  const store = await objm.open('theseus-inferences')
  const cutoff = Date.now() - olderThanMs
  const entries = await store.list()
  let deleted = 0
  for (const entry of entries) {
    if (new Date(entry.mtime).getTime() < cutoff) {
      await store.delete(entry.name)
      deleted++
    }
  }
  return deleted
}

/* ------------------------------------------------------------------ */
/*  Subscribing (pull-based durable consumers)                        */
/* ------------------------------------------------------------------ */

export interface NatsMessage {
  subject: string
  data: unknown
}

export interface SubscribeOptions {
  /** Seconds to wait for an ack before redelivering. Default 30s. */
  ackWaitSeconds?: number
  /**
   * How many times a message may be redelivered before it's given up on.
   * On the last attempt, a failing handler no longer naks — the message is
   * acked (stopping redelivery), published to THESEUS_DLQ for operator
   * inspection/replay, and dropped. Previously this was unbounded, so a
   * poison message would nak forever. Default 5.
   */
  maxDeliver?: number
  /** DLQ subject `kind` token (e.g. 'run-events') for permanently-failed messages. */
  dlqKind?: string
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
  filterSubject: string,
  durableName: string,
  handler: (data: T, subject: string) => Promise<void>,
  signal?: AbortSignal,
  options: SubscribeOptions = {},
): Promise<void> {
  const ackWait = (options.ackWaitSeconds ?? 30) * 1_000_000_000
  const maxDeliver = options.maxDeliver ?? 5
  const dlqKind = options.dlqKind ?? durableName

  // Ensure consumer exists
  try {
    await jsm.consumers.info(stream, durableName)
  } catch {
    await jsm.consumers.add(stream, {
      durable_name: durableName,
      filter_subject: filterSubject,
      deliver_policy: DeliverPolicy.All,
      ack_policy: AckPolicy.Explicit,
      ack_wait: ackWait,
      max_deliver: maxDeliver,
    })
  }

  const consumer = await js.consumers.get(stream, durableName)

  // Process messages in background
  ;(async () => {
    while (!signal?.aborted) {
      try {
        const messages = await consumer.fetch({ max_messages: 1, expires: 5_000 })
        for await (const msg of messages) {
          let data: T | undefined
          try {
            data = msg.json<T>()
            const parentCtx = extractTraceContext(msg.headers)
            await context.with(parentCtx, () =>
              tracer.startActiveSpan(`nats.consume ${msg.subject}`, async (span) => {
                try {
                  await handler(data as T, msg.subject)
                } finally {
                  span.end()
                }
              }),
            )
            msg.ack()
          } catch (err) {
            const deliveryCount = msg.info.deliveryCount
            console.error(
              `[nats] Handler error for ${msg.subject} (attempt ${deliveryCount}/${maxDeliver}):`,
              err,
            )
            if (deliveryCount >= maxDeliver) {
              console.error(`[nats] Giving up on ${msg.subject} after ${deliveryCount} attempts — sending to DLQ.`)
              await publishToDlq(dlqKind, msg.subject, {
                originalSubject: msg.subject,
                payload: data ?? null,
                error: err instanceof Error ? err.message : String(err),
                deliveryCount,
              }).catch((dlqErr) => console.error('[nats] Failed to publish to DLQ:', dlqErr))
              msg.ack()
            } else {
              msg.nak(10_000) // Retry after 10s
            }
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

  console.log(`[nats] Subscribed: stream=${stream}, subject=${filterSubject}, consumer=${durableName}`)
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
    filter_subject: subject.runEventsWildcard(runId),
    ack_policy: AckPolicy.Explicit,
    ...(afterSeq
      ? { deliver_policy: DeliverPolicy.StartSequence, opt_start_seq: afterSeq + 1 }
      : { deliver_policy: DeliverPolicy.All }),
    inactive_threshold: 5 * 60 * 1_000_000_000, // 5 minutes
  })
  return js.consumers.get('THESEUS_EVENTS', info.name)
}
