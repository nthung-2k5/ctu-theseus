/**
 * NATS JetStream client for the ElysiaJS gateway.
 *
 * Provides connection management, stream provisioning, and publish/subscribe
 * helpers.
 */

import {
  AckPolicy,
  connect,
  DeliverPolicy,
  type JetStreamClient,
  type JetStreamManager,
  type NatsConnection,
  RetentionPolicy,
  StringCodec,
} from 'nats'

const sc = StringCodec()

let nc: NatsConnection
let js: JetStreamClient
let jsm: JetStreamManager

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
  js = nc.jetstream()
  jsm = await nc.jetstreamManager()

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
export async function publish(subject: string, data: unknown): Promise<void> {
  const payload = sc.encode(JSON.stringify(data))
  const ack = await js.publish(subject, payload)
  console.log(`[nats] Published to ${subject} (stream=${ack.stream}, seq=${ack.seq})`)
}

/**
 * Publish a training task.
 */
export async function publishTrainTask(projectId: string, data: unknown): Promise<void> {
  await publish(`theseus.tasks.train.${projectId}`, data)
}

/**
 * Publish an inference task.
 */
export async function publishInferenceTask(projectId: string, data: unknown): Promise<void> {
  await publish(`theseus.tasks.inference.${projectId}`, data)
}

/**
 * Publish an export task.
 */
export async function publishExportTask(projectId: string, data: unknown): Promise<void> {
  await publish(`theseus.tasks.export.${projectId}`, data)
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
export async function subscribe(
  stream: string,
  subject: string,
  durableName: string,
  handler: (data: Record<string, unknown>, subject: string) => Promise<void>,
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
            const data = JSON.parse(sc.decode(msg.data)) as Record<string, unknown>
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
