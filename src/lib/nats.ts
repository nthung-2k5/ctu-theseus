/**
 * NATS JetStream client for the ElysiaJS gateway.
 *
 * Provides connection management, stream provisioning, and publish/subscribe
 * helpers.
 */

import {
  AckPolicy,
  DeliverPolicy,
  type JetStreamClient,
  type JetStreamManager,
  jetstream,
  jetstreamManager,
  RetentionPolicy,
} from '@nats-io/jetstream'
import type { NatsConnection } from '@nats-io/nats-core'
import { Objm } from '@nats-io/obj'
import { connect } from '@nats-io/transport-node'

let nc: NatsConnection
let js: JetStreamClient
let jsm: JetStreamManager
let objm: Objm

const NATS_URL = process.env.NATS_URL ?? 'nats://localhost:4222'

/* ------------------------------------------------------------------ */
/*  Connection & Stream Setup                                         */
/* ------------------------------------------------------------------ */

/**
 * Connect to NATS and provision JetStream streams.
 * Call once at gateway startup.
 */
export async function initNats(): Promise<void> {
  console.log(`[nats] Connecting to ${NATS_URL}...`)

  nc = await connect({ servers: NATS_URL })
  js = jetstream(nc)
  jsm = await jetstreamManager(nc)
  objm = new Objm(js)

  // Provision streams (idempotent — creates if missing, updates if exists)
  const streams = [
    {
      name: 'TASKS',
      subjects: ['theseus.tasks.>'],
      retention: RetentionPolicy.Workqueue,
      max_age: 24 * 3600 * 1_000_000_000, // 24h in nanoseconds
    },
    {
      name: 'RESULTS',
      subjects: ['theseus.results.>'],
      retention: RetentionPolicy.Limits,
      max_age: 7 * 24 * 3600 * 1_000_000_000, // 7 days
    },
    {
      name: 'PROGRESS',
      subjects: ['theseus.progress.>'],
      retention: RetentionPolicy.Limits,
      max_age: 3600 * 1_000_000_000, // 1 hour
    },
    {
      name: 'COMMANDS',
      subjects: ['theseus.commands.>'],
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
 * Publish a JSON message to a JetStream subject.
 */
async function publish(subject: string, data: unknown): Promise<void> {
  const ack = await js.publish(subject, JSON.stringify(data))
  console.log(`[nats] Published to ${subject} (stream=${ack.stream}, seq=${ack.seq})`)
}

/**
 * Publish a training task.
 */
export async function publishTrainTask(data: unknown): Promise<void> {
  await publish(`theseus.tasks.train`, data)
}

/**
 * Publish an inference task.
 */
export interface InferencePayload {
  uploadKey: string
  threshold: number
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

export async function requestInferenceTask(modelId: string, data: InferencePayload): Promise<InferenceResponse> {
  const msg = await nc.request(`theseus.inference.${modelId}`, JSON.stringify(data))
  return await msg.json()
}

/**
 * Publish an export task.
 */
export async function publishExportTask(data: unknown): Promise<void> {
  await publish(`theseus.tasks.export`, data)
}

/**
 * Publish an abort command for a training run.
 */
export async function publishAbortCommand(runId: string): Promise<void> {
  await publish(`theseus.commands.train.${runId}`, {
    command: 'abort',
    run_id: runId,
  })
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
/*  Subscribing (push-based ordered consumers)                         */
/* ------------------------------------------------------------------ */

export interface NatsMessage {
  subject: string
  data: unknown
}

/**
 * Subscribe to a JetStream subject with a durable consumer.
 * The handler is called for each message. Messages are auto-acked after
 * successful handler execution.
 */
export async function subscribe<T>(
  stream: string,
  subject: string,
  durableName: string,
  handler: (data: T, subject: string) => Promise<void>,
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
    while (true) {
      try {
        const messages = await consumer.fetch({ max_messages: 1, expires: 5_000 })
        for await (const msg of messages) {
          try {
            const data = msg.json<T>()
            await handler(data, msg.subject)
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
  })()

  console.log(`[nats] Subscribed: stream=${stream}, subject=${subject}, consumer=${durableName}`)
}
