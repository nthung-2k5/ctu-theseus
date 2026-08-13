/**
 * AI Microservice Integration Layer
 *
 * Communicates with the AI worker service via NATS JetStream.
 *
 * - Model registry is loaded from the AI worker via NATS request-reply at startup
 * - Training/inference/export tasks are published as NATS messages
 * - Run events (status/metric/log) are consumed from the single THESEUS_EVENTS
 *   stream via one durable consumer that persists to Postgres
 */

import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { trainingMetrics, trainingRuns } from '@server/db/schema'
import type { ProjectTask, TrainingStatuses } from '@server/lib/enums'
import { compileLudwigConfig, serializeLudwigConfig, type TrainerSelections } from '@server/lib/ludwig'
import { readSnapshotManifest } from '@server/lib/snapshot'
import { snapshotParquetKey, trainingConfigKey, trainingResultsPrefix, uploadFile } from '@server/lib/storage'
import { getTaskDescriptor } from '@server/lib/tasks'
import { record } from '@server/lib/telemetry'
import { eq } from 'drizzle-orm'
import { publishAbortCommand, publishExportTask, publishTrainTask, subscribe } from './nats'

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

export interface QueueTrainingParams {
  projectId: string
  name: string
  task: ProjectTask
  datasetVersionId: string
  trainerSelections?: TrainerSelections
}

export type QueueTrainingResult =
  | { ok: true; run: typeof trainingRuns.$inferSelect }
  | { ok: false; code: 404 | 409 | 400; message: string }

/**
 * Compile the Ludwig config for a run, upload it, persist the run row, and
 * dispatch the task over NATS — in that order, so an invalid configuration
 * or a not-ready dataset version surfaces as a 4xx here rather than as a
 * silent worker failure discovered later.
 */
export async function queueTraining(params: QueueTrainingParams): Promise<QueueTrainingResult> {
  return record('train.dispatch', async (span): Promise<QueueTrainingResult> => {
    span.setAttributes({ 'theseus.project_id': params.projectId, 'theseus.task': params.task })

    const version = await db.query.datasetVersions.findFirst({ where: { id: params.datasetVersionId } })
    if (!version) return { ok: false, code: 404, message: 'Dataset version not found' }
    if (version.status !== 'ready') {
      return { ok: false, code: 409, message: `Dataset version is not ready for training (status: ${version.status})` }
    }

    const descriptor = getTaskDescriptor(params.task)
    let configYaml: string
    let ludwigConfig: ReturnType<typeof compileLudwigConfig>
    try {
      const ctx = await readSnapshotManifest(params.datasetVersionId)
      ludwigConfig = compileLudwigConfig(descriptor, ctx, params.trainerSelections ?? {})
      configYaml = serializeLudwigConfig(ludwigConfig)
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      return { ok: false, code: 400, message: `Failed to compile Ludwig config: ${message}` }
    }

    const runId = crypto.randomUUID()
    span.setAttribute('theseus.run_id', runId)
    const configKey = trainingConfigKey(runId)
    await uploadFile(CONSTANTS.BUCKET_TRAINING, configKey, new TextEncoder().encode(configYaml), 'application/yaml')

    const [run] = await db
      .insert(trainingRuns)
      .values({
        id: runId,
        projectId: params.projectId,
        name: params.name,
        datasetVersionId: params.datasetVersionId,
        hyperparameters: params.trainerSelections ?? {},
        ludwigConfig,
        configKey,
        status: 'queued',
        updatedAt: new Date(),
      })
      .returning()

    try {
      await publishTrainTask(runId, {
        runId,
        projectId: params.projectId,
        datasetVersionId: params.datasetVersionId,
        configKey,
        datasetKey: snapshotParquetKey(params.datasetVersionId),
        outputPrefix: trainingResultsPrefix(runId),
      })
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : String(e)
      await db
        .update(trainingRuns)
        .set({ status: 'failed', failedMessage: message, completedAt: new Date() })
        .where(eq(trainingRuns.id, runId))
      console.error(`Training dispatch failed: ${e}`)
    }

    return { ok: true, run }
  })
}

/**
 * Stop a running training task by publishing an abort command to NATS.
 */
export async function stopTraining(runId: string) {
  await publishAbortCommand(runId)
}

/* ------------------------------------------------------------------ */
/*  Export                                                            */
/* ------------------------------------------------------------------ */

/**
 * Dispatch a model-export job for a succeeded run. `jobId` doubles as the
 * NATS subject token and, once the worker publishes it, the export key
 * (`exportKey(runId, format)` — one run, one artifact per format).
 */
export async function dispatchExport(runId: string, format: 'onnx' | 'torchscript'): Promise<string> {
  const jobId = crypto.randomUUID()
  await publishExportTask(jobId, { jobId, runId, format })
  return jobId
}

/* ------------------------------------------------------------------ */
/*  NATS Event Consumer                                               */
/* ------------------------------------------------------------------ */

/**
 * A run event as published by the worker (RunEventSchema in src/lib/schema.ts).
 * Tagged union over `kind`. `status` values match the trainingRuns.status
 * enum 1:1 — no derivation/translation needed on this side anymore.
 */
type RunEvent =
  | { kind: 'status'; runId: string; ts: string; status: (typeof TrainingStatuses)[number]; message?: string }
  | {
      kind: 'metric'
      runId: string
      ts: string
      epoch: number
      split: 'train' | 'validation' | 'test'
      metrics: Record<string, number>
    }
  | { kind: 'log'; runId: string; ts: string; level: 'info' | 'warn' | 'error'; line: string }

const TERMINAL_STATUSES = new Set<(typeof TrainingStatuses)[number]>(['succeeded', 'failed', 'canceled'])

/**
 * Start the durable NATS consumer that persists run events to Postgres.
 * Called once at gateway startup after initNats().
 *
 * This is the single source of truth in the DB; the SSE route (next phase)
 * reads live events directly off the JetStream stream instead, so a
 * refreshing browser can replay history without round-tripping Postgres.
 */
export async function startNatsConsumers(signal?: AbortSignal): Promise<void> {
  await subscribe<RunEvent>(
    'THESEUS_EVENTS',
    'theseus.event.run.*.>',
    'gateway-run-events',
    async (event) => {
      switch (event.kind) {
        case 'status': {
          const patch: Partial<typeof trainingRuns.$inferInsert> = { status: event.status, heartbeatAt: new Date() }
          if (event.status === 'running') patch.startedAt = new Date()
          if (TERMINAL_STATUSES.has(event.status)) {
            patch.completedAt = new Date()
            if (event.status === 'failed' && event.message) patch.failedMessage = event.message
          }
          await db.update(trainingRuns).set(patch).where(eq(trainingRuns.id, event.runId))
          break
        }
        case 'metric': {
          for (const [metricName, metricValue] of Object.entries(event.metrics)) {
            await db
              .insert(trainingMetrics)
              .values({ trainingRunId: event.runId, epoch: event.epoch, split: event.split, metricName, metricValue })
              .onConflictDoUpdate({
                target: [
                  trainingMetrics.trainingRunId,
                  trainingMetrics.epoch,
                  trainingMetrics.split,
                  trainingMetrics.metricName,
                ],
                set: { metricValue },
              })
          }
          // NOTE: bestEpoch tracking (vs. just the latest epoch) needs to
          // know which metric the task is optimizing for — deferred to the
          // Ludwig compiler phase, which has that context.
          await db
            .update(trainingRuns)
            .set({ status: 'running', heartbeatAt: new Date() })
            .where(eq(trainingRuns.id, event.runId))
          break
        }
        case 'log': {
          // Logs aren't persisted to Postgres — the SSE route replays them
          // directly from the JetStream stream. Just mark the run alive.
          await db.update(trainingRuns).set({ heartbeatAt: new Date() }).where(eq(trainingRuns.id, event.runId))
          break
        }
      }
    },
    signal,
  )

  console.log('[microservice] NATS run-events consumer started.')
}
