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
import { inferenceJobs, modelExports, runEvaluations, sweeps, trainingMetrics, trainingRuns } from '@server/db/schema'
import type { EvaluationSplit, ExportFormat, ProjectTask, SweepStrategy, TrainingStatuses } from '@server/lib/enums'
import { enqueueAssembly } from '@server/lib/export/queue'
import { compileLudwigConfig, serializeLudwigConfig, type TrainerSelections } from '@server/lib/ludwig'
import { instrumentNatsHandler, trainingRunTerminalCount } from '@server/lib/metrics'
import { readSnapshotManifest } from '@server/lib/snapshot'
import {
  downloadFile,
  snapshotParquetKey,
  trainingConfigKey,
  trainingResultsPrefix,
  uploadFile,
} from '@server/lib/storage'
import { SUBJECT_WILDCARDS } from '@server/lib/subjects'
import { expandSweep, type SweepSearchSpace, validateSearchSpace } from '@server/lib/sweep'
import { getTaskDescriptor } from '@server/lib/tasks'
import { record } from '@server/lib/telemetry'
import { and, eq, inArray, isNull, lt, notInArray, or } from 'drizzle-orm'
import type { InferenceResponse } from './nats'
import { publishAbortCommand, publishExportTask, publishTrainTask, subscribe, sweepStaleInferenceUploads } from './nats'

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

export interface QueueTrainingParams {
  projectId: string
  name: string
  task: ProjectTask
  datasetVersionId: string
  trainerSelections?: TrainerSelections
  /** Set only when this run is one trial of a sweep — see queueSweep below. */
  sweepId?: string
  trialIndex?: number
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

    const runId = Bun.randomUUIDv7()
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
        sweepId: params.sweepId,
        trialIndex: params.trialIndex,
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
/*  Sweeps                                                             */
/* ------------------------------------------------------------------ */

export interface QueueSweepParams {
  projectId: string
  name: string
  task: ProjectTask
  datasetVersionId: string
  searchSpace: SweepSearchSpace
  strategy: SweepStrategy
  maxTrials: number
}

export type QueueSweepResult =
  | { ok: true; sweep: typeof sweeps.$inferSelect; trials: (typeof trainingRuns.$inferSelect)[] }
  | { ok: false; code: 404 | 409 | 400; message: string }

/**
 * Expand a search space into trials and dispatch each one through the
 * ordinary queueTraining path — a sweep has no execution engine of its own
 * (see server/lib/sweep.ts's module docstring). Trials dispatch serially,
 * in trial order: THESEUS_TASKS is a workqueue stream, so with one worker
 * replica they still train one at a time regardless, and serial dispatch
 * means a config error on trial 3 doesn't leave trials 4..N half-created if
 * something goes wrong before it — every trial that queueTraining accepted
 * before the failure is left running (an early return here would abandon
 * them with no sweep to belong to).
 */
export async function queueSweep(params: QueueSweepParams): Promise<QueueSweepResult> {
  return record('sweep.dispatch', async (span): Promise<QueueSweepResult> => {
    span.setAttributes({ 'theseus.project_id': params.projectId, 'theseus.task': params.task })

    const version = await db.query.datasetVersions.findFirst({ where: { id: params.datasetVersionId } })
    if (!version) return { ok: false, code: 404, message: 'Dataset version not found' }
    if (version.status !== 'ready') {
      return { ok: false, code: 409, message: `Dataset version is not ready for training (status: ${version.status})` }
    }

    const validationError = validateSearchSpace(params.searchSpace, params.maxTrials)
    if (validationError) return { ok: false, code: 400, message: validationError }

    const trialSelections = expandSweep(params.searchSpace, params.strategy, params.maxTrials)

    const [sweep] = await db
      .insert(sweeps)
      .values({
        projectId: params.projectId,
        datasetVersionId: params.datasetVersionId,
        name: params.name,
        searchSpace: params.searchSpace,
        strategy: params.strategy,
        maxTrials: params.maxTrials,
        status: 'running',
      })
      .returning()

    const trials: (typeof trainingRuns.$inferSelect)[] = []
    for (const [trialIndex, trainerSelections] of trialSelections.entries()) {
      const result = await queueTraining({
        projectId: params.projectId,
        name: `${params.name} — trial ${trialIndex + 1}`,
        task: params.task,
        datasetVersionId: params.datasetVersionId,
        trainerSelections,
        sweepId: sweep.id,
        trialIndex,
      })
      // A single trial failing to compile (e.g. an invalid encoderId that
      // slipped past validateSearchSpace) shouldn't abort the whole sweep —
      // queueTraining already turns that into a 'failed' run row on its own,
      // so it just shows up as a failed trial in the leaderboard.
      if (result.ok) trials.push(result.run)
    }

    return { ok: true, sweep, trials }
  })
}

/**
 * Cancel a sweep and every one of its trials that hasn't already finished.
 */
export async function cancelSweep(sweepId: string): Promise<void> {
  const trials = await db.query.trainingRuns.findMany({
    where: { sweepId, status: { in: ['queued', 'running'] } },
    columns: { id: true },
  })
  for (const trial of trials) await stopTraining(trial.id)
  await db
    .update(trainingRuns)
    .set({ status: 'canceled', completedAt: new Date() })
    .where(and(eq(trainingRuns.sweepId, sweepId), inArray(trainingRuns.status, ['queued', 'running'])))
  await db.update(sweeps).set({ status: 'canceled' }).where(eq(sweeps.id, sweepId))
}

/**
 * Lazily bring a sweep's status up to date: 'running' becomes 'completed'
 * once every trial has reached a terminal status on its own. Mirrors
 * routes/export.ts's reconcileConverting — computed on read rather than
 * via a reaper, since nothing time-sensitive depends on a sweep flipping to
 * 'completed' the instant its last trial finishes.
 */
export async function reconcileSweepStatus(sweep: typeof sweeps.$inferSelect): Promise<typeof sweeps.$inferSelect> {
  if (sweep.status !== 'running') return sweep

  const nonTerminal = await db.query.trainingRuns.findFirst({
    where: { sweepId: sweep.id, status: { in: ['queued', 'running'] } },
    columns: { id: true },
  })
  if (nonTerminal) return sweep

  const [updated] = await db
    .update(sweeps)
    .set({ status: 'completed' })
    .where(and(eq(sweeps.id, sweep.id), eq(sweeps.status, 'running')))
    .returning()
  return updated ?? sweep
}

/* ------------------------------------------------------------------ */
/*  Export                                                            */
/* ------------------------------------------------------------------ */

/**
 * Dispatch a model-export job for a succeeded run. `jobId` doubles as the
 * NATS subject token and, once the worker publishes it, the export key
 * (`exportKey(runId, format)` — one run, one artifact per format).
 *
 * Looks up the run's source snapshot so the worker can verify the export
 * against one real test-split row (see server/lib/export/bundle.ts and
 * ai_service/tasks/export.py) — best-effort: a missing/deleted snapshot
 * just means no `expected.json` gets written, not a failed export.
 */
export async function dispatchExport(runId: string, format: ExportFormat): Promise<string> {
  const jobId = Bun.randomUUIDv7()
  const run = await db.query.trainingRuns.findFirst({ where: { id: runId } })
  const datasetKey = run ? snapshotParquetKey(run.datasetVersionId) : undefined
  await publishExportTask(jobId, { jobId, runId, format, datasetKey })
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
  | {
      kind: 'export'
      runId: string
      ts: string
      jobId: string
      status: 'success' | 'failed'
      format?: ExportFormat
      exportKey?: string
      error?: string
    }
  | {
      kind: 'evaluation'
      runId: string
      ts: string
      status: 'success' | 'failed'
      split?: EvaluationSplit
      reportKey?: string
      predictionsKey?: string
      headlineMetric?: number
      error?: string
    }

const TERMINAL_STATUSES = new Set<(typeof TrainingStatuses)[number]>(['succeeded', 'failed', 'canceled'])
const TERMINAL_STATUS_LIST = [...TERMINAL_STATUSES]

/**
 * Guards every run-status write against resurrecting a finished run.
 *
 * THESEUS_EVENTS is at-least-once, so a `running` or `metric` event can be
 * redelivered *after* the run's terminal event was already processed — which
 * previously flipped a succeeded run back to `running` and stamped a fresh
 * `completedAt`. The reaper (below) creates the same hazard from the other
 * direction: it marks a silent run `failed`, and a late event would undo that.
 * Terminal is terminal; only the cancel route and the reaper move a run there.
 */
const notTerminal = (runId: string) =>
  and(eq(trainingRuns.id, runId), notInArray(trainingRuns.status, TERMINAL_STATUS_LIST))

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
    SUBJECT_WILDCARDS.allRunEvents,
    'gateway-run-events',
    instrumentNatsHandler<RunEvent>(
      'run-events',
      (event) => event.kind,
      async (event) => {
        switch (event.kind) {
          case 'status': {
            const patch: Partial<typeof trainingRuns.$inferInsert> = { status: event.status, heartbeatAt: new Date() }
            if (event.status === 'running') patch.startedAt = new Date()
            if (TERMINAL_STATUSES.has(event.status)) {
              patch.completedAt = new Date()
              if (event.status === 'failed' && event.message) patch.failedMessage = event.message
            }
            // A terminal status may be written once. 'canceled' in particular is
            // set by the cancel route ahead of the worker confirming the abort
            // (see routes/training.ts), and an in-flight event must not undo it.
            const [updated] = await db
              .update(trainingRuns)
              .set(patch)
              .where(notTerminal(event.runId))
              .returning({ id: trainingRuns.id })
            if (updated && TERMINAL_STATUSES.has(event.status)) {
              trainingRunTerminalCount.add(1, { status: event.status })
            }
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

            const patch: Partial<typeof trainingRuns.$inferInsert> = { status: 'running', heartbeatAt: new Date() }

            // Ludwig always trains to minimize loss regardless of task type
            // (classification, regression, ...), so "lowest validation loss
            // so far" is a task-agnostic definition of "best epoch" that
            // doesn't need per-task optimization-metric metadata.
            if (event.split === 'validation' && 'loss' in event.metrics) {
              const run = await db.query.trainingRuns.findFirst({
                where: { id: event.runId },
                columns: { bestEpoch: true },
              })
              const bestSoFar =
                run?.bestEpoch != null
                  ? await db.query.trainingMetrics.findFirst({
                      where: {
                        trainingRunId: event.runId,
                        epoch: run.bestEpoch,
                        split: 'validation',
                        metricName: 'loss',
                      },
                    })
                  : null
              if (!bestSoFar || event.metrics.loss < bestSoFar.metricValue) {
                patch.bestEpoch = event.epoch
              }
            }

            // Same race as the 'status' case: a metric event that arrives (or is
            // redelivered) after the run finished must not flip it back to
            // 'running'.
            await db.update(trainingRuns).set(patch).where(notTerminal(event.runId))
            break
          }
          case 'log': {
            // Logs aren't persisted to Postgres — the SSE route replays them
            // directly from the JetStream stream. Just mark the run alive.
            await db.update(trainingRuns).set({ heartbeatAt: new Date() }).where(notTerminal(event.runId))
            break
          }
          case 'export': {
            if (event.status === 'failed') {
              await db
                .update(modelExports)
                .set({ status: 'failed', failedMessage: event.error ?? 'Model conversion failed' })
                .where(and(eq(modelExports.conversionJobId, event.jobId), eq(modelExports.status, 'converting')))
              break
            }
            // Guarded UPDATE: zero rows means this jobId was already promoted
            // (redelivery, or the lazy-reconcile check in routes/export.ts got
            // there first) — either way, only enqueue once.
            const promoted = await db
              .update(modelExports)
              .set({ status: 'assembling' })
              .where(and(eq(modelExports.conversionJobId, event.jobId), eq(modelExports.status, 'converting')))
              .returning({ id: modelExports.id })
            for (const row of promoted) enqueueAssembly(row.id)
            break
          }
          case 'evaluation': {
            // One row per run (runId is the primary key) — an upsert, so
            // redelivery just rewrites the same/latest data rather than
            // needing a guarded UPDATE like the 'export' case above.
            if (event.status === 'failed') {
              await db
                .insert(runEvaluations)
                .values({ runId: event.runId, status: 'failed', failedMessage: event.error ?? 'Evaluation failed' })
                .onConflictDoUpdate({
                  target: runEvaluations.runId,
                  set: { status: 'failed', failedMessage: event.error ?? 'Evaluation failed', evaluatedAt: new Date() },
                })
              break
            }

            // Bounded document (ai_service caps topErrors and skips the
            // confusion matrix past ~200 classes — see services/evaluate.py),
            // so downloading and parsing it inline stays well within the
            // consumer's ack window.
            let report: Record<string, unknown> | null = null
            if (event.reportKey) {
              try {
                const bytes = await downloadFile(CONSTANTS.BUCKET_TRAINING, event.reportKey)
                report = JSON.parse(new TextDecoder().decode(bytes))
              } catch (e) {
                console.error(`[microservice] Failed to download evaluation report for run ${event.runId}:`, e)
              }
            }
            const overall = (report?.overall ?? null) as { accuracy?: number; macroF1?: number } | null

            await db
              .insert(runEvaluations)
              .values({
                runId: event.runId,
                status: 'success',
                split: event.split,
                reportKey: event.reportKey,
                predictionsKey: event.predictionsKey,
                report,
                accuracy: overall?.accuracy ?? null,
                macroF1: overall?.macroF1 ?? null,
              })
              .onConflictDoUpdate({
                target: runEvaluations.runId,
                set: {
                  status: 'success',
                  split: event.split,
                  reportKey: event.reportKey,
                  predictionsKey: event.predictionsKey,
                  report,
                  accuracy: overall?.accuracy ?? null,
                  macroF1: overall?.macroF1 ?? null,
                  failedMessage: null,
                  evaluatedAt: new Date(),
                },
              })
            break
          }
        }
      },
    ),
    signal,
    // Fast Postgres-write handler — short ack window is fine. Capped
    // retries + DLQ (see lib/nats.ts `subscribe`) replace the old
    // nak-forever behavior for a run event that can never be applied.
    { ackWaitSeconds: 30, maxDeliver: 5, dlqKind: 'run-event' },
  )

  console.log('[microservice] NATS run-events consumer started.')
}

/* ------------------------------------------------------------------ */
/*  Inference job history                                             */
/* ------------------------------------------------------------------ */

/**
 * Durable consumer that persists every terminal inference result to
 * Postgres the moment the worker publishes it, rather than lazily when a
 * client happens to poll (see routes/inference.ts's dispatch route, which
 * inserts the 'pending' row this updates, before publishing the task —
 * so the row always exists before any result for it could possibly arrive).
 * Without this, a result nobody polls for before
 * THESEUS_INFERENCE_RESULTS' 1-hour retention window expires is lost with
 * no trace, since that stream was the only place it ever lived.
 *
 * InferenceResponseSchema's payload never carries the inferenceId (only
 * runId) — it's the subject's own trailing token
 * (`theseus.inference.result.{inferenceId}`), which is why this needs the
 * subject `subscribe` passes alongside the parsed payload, not just the
 * payload itself.
 */
export async function startInferenceResultsConsumer(signal?: AbortSignal): Promise<void> {
  await subscribe<InferenceResponse>(
    'THESEUS_INFERENCE_RESULTS',
    SUBJECT_WILDCARDS.inferenceResultsAll,
    'gateway-inference-results',
    instrumentNatsHandler<InferenceResponse>(
      'inference-results',
      (result) => result.status,
      async (result, subject) => {
        const inferenceId = subject.split('.').pop()
        if (!inferenceId) return

        // Guarded UPDATE: zero rows means this job was already resolved
        // (redelivery) — safe to no-op rather than risk inserting an orphan
        // row with no run scope this consumer has no way to fill in.
        //
        // A batch result is stored as `status: 'success'` too — the DB enum
        // doesn't need its own 'batch' value, since `output.kind` already
        // discriminates a downloadable-file result from an inline one (see
        // routes/inference.ts's poll route).
        const patch =
          result.status === 'success'
            ? { status: 'success' as const, output: result.output, completedAt: new Date() }
            : result.status === 'batch'
              ? {
                  status: 'success' as const,
                  output: { kind: 'batch' as const, resultKey: result.resultKey, rowCount: result.rowCount },
                  completedAt: new Date(),
                }
              : { status: 'failed' as const, error: result.error, completedAt: new Date() }
        await db
          .update(inferenceJobs)
          .set(patch)
          .where(and(eq(inferenceJobs.id, inferenceId), eq(inferenceJobs.status, 'pending')))
      },
    ),
    signal,
    // Fast Postgres-write handler — short ack window is fine, same as the
    // run-events consumer above.
    { ackWaitSeconds: 30, maxDeliver: 5, dlqKind: 'inference-result' },
  )

  console.log('[microservice] NATS inference-results consumer started.')
}

/* ------------------------------------------------------------------ */
/*  Orphaned-run reaper                                               */
/* ------------------------------------------------------------------ */

/**
 * A run with no heartbeat for this long (worker crash/restart, lost NATS
 * messages) is presumed dead.
 *
 * Keep this aligned with the worker's `TASK_ACK_WAIT_SECONDS`
 * (ai_service/services/nats.py). When the two drifted apart — 5 min here vs a
 * 60 min ack_wait there — killing the worker mid-run made the reaper fail the
 * run at T+5min while JetStream only redelivered it at T+60min, and the
 * retrain then resurrected it: users saw failed -> running -> succeeded.
 */
const HEARTBEAT_STALE_MS = 5 * 60 * 1000
/** A `queued` run that never got a heartbeat at all (dispatch never reached the worker) waits longer before being reaped, since queueing delay alone is normal. */
const NEVER_STARTED_STALE_MS = 15 * 60 * 1000
const REAP_INTERVAL_MS = 60 * 1000

/**
 * Periodically fails `queued`/`running` runs whose heartbeat (see the
 * `status`/`metric`/`log` cases above — all three bump it) has gone stale.
 * Without this, a gateway or worker restart mid-run leaves the run stuck
 * "running" forever, since nothing else ever transitions it out.
 */
export function startOrphanReaper(signal?: AbortSignal): void {
  const reap = async () => {
    try {
      const heartbeatCutoff = new Date(Date.now() - HEARTBEAT_STALE_MS)
      const neverStartedCutoff = new Date(Date.now() - NEVER_STARTED_STALE_MS)

      const reaped = await db
        .update(trainingRuns)
        .set({
          status: 'failed',
          failedMessage: 'Run heartbeat timed out — the worker likely crashed or restarted mid-run',
          completedAt: new Date(),
        })
        .where(
          and(
            inArray(trainingRuns.status, ['queued', 'running']),
            or(
              lt(trainingRuns.heartbeatAt, heartbeatCutoff),
              and(isNull(trainingRuns.heartbeatAt), lt(trainingRuns.createdAt, neverStartedCutoff)),
            ),
          ),
        )
        .returning({ id: trainingRuns.id })

      for (const run of reaped) console.warn(`[reaper] Marked orphaned run ${run.id} as failed`)
    } catch (e) {
      console.error('[reaper] Sweep failed:', e)
    }
  }

  const interval = setInterval(reap, REAP_INTERVAL_MS)
  signal?.addEventListener('abort', () => clearInterval(interval))

  console.log('[microservice] Orphan-run reaper started.')
}

/* ------------------------------------------------------------------ */
/*  Stuck-export reaper                                               */
/* ------------------------------------------------------------------ */

/** An export with no progress for this long (lost/DLQ'd export event, worker crash) is presumed dead. */
const EXPORT_STALE_MS = 15 * 60 * 1000

/**
 * Mirrors `startOrphanReaper`, for `modelExports` instead of `trainingRuns`:
 * a row stuck in `converting` (waiting on the worker's export event) or
 * `assembling` (waiting on the gateway's own zip job) with no `updatedAt`
 * progress for a while is presumed dead and flipped to `failed`. Without
 * this, a lost/DLQ'd `export` run-event or a crashed assembly job left a
 * row stuck forever — nothing else ever moved it out of `converting`.
 */
export function startExportReaper(signal?: AbortSignal): void {
  const reap = async () => {
    try {
      const staleCutoff = new Date(Date.now() - EXPORT_STALE_MS)

      const reaped = await db
        .update(modelExports)
        .set({ status: 'failed', failedMessage: 'Export timed out — no progress from the worker or assembly job' })
        .where(and(inArray(modelExports.status, ['converting', 'assembling']), lt(modelExports.updatedAt, staleCutoff)))
        .returning({ id: modelExports.id })

      for (const exp of reaped) console.warn(`[reaper] Marked stuck export ${exp.id} as failed`)
    } catch (e) {
      console.error('[reaper] Export sweep failed:', e)
    }
  }

  const interval = setInterval(reap, REAP_INTERVAL_MS)
  signal?.addEventListener('abort', () => clearInterval(interval))

  console.log('[microservice] Export reaper started.')
}

/* ------------------------------------------------------------------ */
/*  Stale inference-upload reaper                                     */
/* ------------------------------------------------------------------ */

/** An inference upload older than this was either consumed-but-not-cleaned-up or abandoned mid-request. */
const INFERENCE_UPLOAD_STALE_MS = 60 * 60 * 1000
const INFERENCE_UPLOAD_SWEEP_INTERVAL_MS = 15 * 60 * 1000

/**
 * The worker deletes an upload itself once its inference job succeeds or
 * permanently fails (`nats_service.delete_upload` in
 * ai_service/tasks/inference.py), but a crash or a job whose message never
 * reaches a worker leaves the object behind — and unlike S3, nothing else
 * ever sweeps the `theseus-inferences` object store (`server/lib/cleanup.ts`
 * only covers the three S3 buckets). Without this, uploads accumulate
 * forever.
 */
export function startInferenceUploadReaper(signal?: AbortSignal): void {
  const reap = async () => {
    try {
      const deleted = await sweepStaleInferenceUploads(INFERENCE_UPLOAD_STALE_MS)
      if (deleted > 0) console.warn(`[reaper] Swept ${deleted} stale inference upload(s)`)
    } catch (e) {
      console.error('[reaper] Inference-upload sweep failed:', e)
    }
  }

  const interval = setInterval(reap, INFERENCE_UPLOAD_SWEEP_INTERVAL_MS)
  signal?.addEventListener('abort', () => clearInterval(interval))

  console.log('[microservice] Inference-upload reaper started.')
}
