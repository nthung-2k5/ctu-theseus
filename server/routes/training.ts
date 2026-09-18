import CONSTANTS from '@schema/constants.json'
import { db } from '@server/db'
import { trainingRuns } from '@server/db/schema'
import { cleanupRunStorage } from '@server/lib/cleanup'
import { queueTraining, stopTraining } from '@server/lib/microservice'
import { createRunEventsConsumer } from '@server/lib/nats'
import { fileExists, getDownloadUrl, trainingLogsKey } from '@server/lib/storage'
import { and, eq, inArray } from 'drizzle-orm'
import { Elysia, sse, status, t } from 'elysia'
import { betterAuth } from './auth'

/** One misclassified row, as ai_service/services/evaluate.py's `_top_errors` writes it into report.json's `topErrors` array. */
interface TopError {
  itemId: string
  actual: string
  predicted: string
  confidence: number | null
}

const ERRORS_PER_PAGE = 50

export const trainingRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  /* ── List training runs for a project ── */
  .get(
    '/projects/:projectId/runs',
    async ({ params }) => {
      const runs = await db.query.trainingRuns.findMany({
        where: {
          projectId: params.projectId,
        },
        columns: {
          id: true,
          name: true,
          status: true,
          failedMessage: true,
          completedAt: true,
          startedAt: true,
          createdAt: true,
          datasetVersionId: true,
        },
        // Denormalized accuracy/macroF1 (see server/db/schema.ts's
        // runEvaluations comment) so the run list and comparison view can
        // show/sort on them without a per-run round trip.
        with: { evaluation: { columns: { status: true, accuracy: true, macroF1: true } } },
        orderBy: { createdAt: 'desc' },
      })

      // `columns` above already restricts the shape, so re-mapping added
      // nothing — it only silently dropped `failedMessage` and `completedAt`,
      // which the query was paying to select anyway and which the run list
      // needs to explain a failure.
      return { runs }
    },
    { projectBelongToUser: true },
  )
  /* ── Get a single training run with its metrics ── */
  .get(
    '/runs/:runId',
    async ({ run }) => {
      const full = await db.query.trainingRuns.findFirst({
        where: { id: run.id },
        with: {
          metrics: true,
          datasetVersion: {
            with: {
              dataset: {
                columns: { projectId: true, modality: true },
              },
            },
          },
        },
      })
      if (!full) return status(404, 'Training run not found')

      return {
        run: {
          id: full.id,
          name: full.name,
          status: full.status,
          hyperparameters: full.hyperparameters,
          failedMessage: full.failedMessage,
          startedAt: full.startedAt,
          completedAt: full.completedAt,
          createdAt: full.createdAt,
          datasetVersion: full.datasetVersion,
          metrics: full.metrics,
        },
      }
    },
    { runBelongToUser: true },
  )
  /* ── Start a new training run ── */
  .post(
    '/projects/:projectId/train',
    async ({ params, body, project }) => {
      // Validate the dataset version exists and belongs to this project
      const version = await db.query.datasetVersions.findFirst({
        where: { id: body.datasetVersionId },
        with: {
          dataset: {
            columns: { projectId: true },
          },
        },
      })

      if (!version) return status(404, 'Dataset version not found')
      if (version.dataset.projectId !== params.projectId) {
        return status(400, 'Dataset version does not belong to this project')
      }

      const result = await queueTraining({
        projectId: params.projectId,
        name: body.name,
        task: project.task,
        datasetVersionId: body.datasetVersionId,
        trainerSelections: body.hyperparameters,
      })

      if (!result.ok) return status(result.code, result.message)
      return { run: result.run }
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        name: t.String({ minLength: 1, maxLength: 255 }),
        datasetVersionId: t.String({ format: 'uuid' }),
        hyperparameters: t.Optional(t.Any()),
      }),
    },
  )
  /* ── Cancel a queued or running training run ── */
  .post(
    '/runs/:runId/cancel',
    async ({ run }) => {
      // Only an in-flight run can be canceled. Without this guard, cancelling
      // an already-succeeded run flipped it to 'canceled' — which permanently
      // blocks inference (routes/inference.ts) and export (routes/export.ts)
      // for a model that trained fine, and the terminal-status guard in
      // microservice.ts means no later event can move it back.
      if (run.status !== 'queued' && run.status !== 'running') {
        return status(409, `Cannot cancel a run with status '${run.status}' — it has already finished`)
      }

      // Publish abort command via NATS
      await stopTraining(run.id)
      // Scope the write the same way, so a run that finished between the read
      // above and this update isn't clobbered.
      await db
        .update(trainingRuns)
        .set({ status: 'canceled', completedAt: new Date() })
        .where(and(eq(trainingRuns.id, run.id), inArray(trainingRuns.status, ['queued', 'running'])))
      return status(204)
    },
    { runBelongToUser: true },
  )
  /* ── Delete a training run, regardless of status ── */
  .delete(
    '/runs/:runId',
    async ({ run }) => {
      // An in-flight run still has a worker training it — stop that job before
      // tearing down its storage and DB row, otherwise the worker keeps
      // running against a run that no longer exists.
      if (run.status === 'queued' || run.status === 'running') {
        await stopTraining(run.id)
      }
      // S3 cleanup before the DB delete, same ordering as project/version deletes.
      await cleanupRunStorage(run.id)
      await db.delete(trainingRuns).where(eq(trainingRuns.id, run.id))
      return status(204)
    },
    { runBelongToUser: true },
  )
  /* ── Get training status with latest metrics (short-poll fallback) ── */
  .get(
    '/runs/:runId/status',
    async ({ run }) => {
      const latestEpochRow = await db.query.trainingMetrics.findFirst({
        where: { trainingRunId: run.id },
        orderBy: { epoch: 'desc' },
        columns: { epoch: true },
      })

      // All metric rows (every split/name) for the most recent epoch
      const latestMetrics = latestEpochRow
        ? await db.query.trainingMetrics.findMany({
            where: { trainingRunId: run.id, epoch: latestEpochRow.epoch },
          })
        : []

      // Extract total epochs from hyperparameters if available
      const hyperparams = run.hyperparameters as Record<string, unknown> | null
      const epochsTotal =
        (hyperparams?.epochs as number) ?? (hyperparams?.schedule as Record<string, unknown>)?.epochs ?? null

      return {
        status: run.status,
        epochsTotal,
        latestMetrics,
      }
    },
    { runBelongToUser: true },
  )
  /* ── Live run events (SSE) — status/metric/log, replayable via Last-Event-ID ── */
  .get(
    '/runs/:runId/events',
    async function* ({ run, request }) {
      const lastEventId = request.headers.get('last-event-id')
      const afterSeq = lastEventId ? Number.parseInt(lastEventId, 10) : undefined

      const consumer = await createRunEventsConsumer(run.id, Number.isFinite(afterSeq) ? afterSeq : undefined)
      try {
        const messages = await consumer.consume()
        for await (const msg of messages) {
          const data = msg.json<{ kind: string }>()
          yield sse({ id: String(msg.seq), event: data.kind, data })
          msg.ack()
        }
      } finally {
        await consumer.delete()
      }
    },
    { runBelongToUser: true },
  )
  /* ── Evaluation report (confusion matrix / per-class stats / regression metrics) for a finished run ── */
  .get(
    '/runs/:runId/evaluation',
    async ({ run }) => {
      const evaluation = await db.query.runEvaluations.findFirst({ where: { runId: run.id } })
      if (!evaluation) return status(404, 'No evaluation report for this run yet')
      return { evaluation }
    },
    { runBelongToUser: true },
  )
  /* ── Paginated misclassified rows from the evaluation report, joined back to their pool item ── */
  .get(
    '/runs/:runId/evaluation/errors',
    async ({ run, query }) => {
      const evaluation = await db.query.runEvaluations.findFirst({ where: { runId: run.id } })
      if (evaluation?.status !== 'success' || !evaluation.report) {
        return status(404, 'No evaluation report for this run yet')
      }

      const report = evaluation.report as { topErrors?: TopError[] }
      let errors = report.topErrors ?? []

      // classId filters to rows whose ACTUAL label is that class — resolved
      // to a name first since topErrors stores Ludwig's own idx2str-derived
      // label strings, never a Postgres class id (see services/evaluate.py's
      // module docstring for why: label_classes has no idea which index
      // Ludwig assigned to which class).
      if (query.classId) {
        const cls = await db.query.labelClasses.findFirst({ where: { classId: query.classId } })
        if (!cls) return status(400, 'Unknown label class')
        errors = errors.filter((e) => e.actual === cls.name)
      }

      const page = Math.max(1, query.page ?? 1)
      const total = errors.length
      const pageErrors = errors.slice((page - 1) * ERRORS_PER_PAGE, page * ERRORS_PER_PAGE)

      const itemIds = pageErrors.map((e) => e.itemId)
      const items = itemIds.length
        ? await db.query.datasetItems.findMany({ where: { id: { in: itemIds } }, with: { textFeatures: true } })
        : []
      const itemById = new Map(items.map((i) => [i.id, i]))

      const rows = await Promise.all(
        pageErrors.map(async (e) => {
          const item = itemById.get(e.itemId)
          return {
            ...e,
            item: item
              ? {
                  id: item.id,
                  text: item.textFeatures?.rawText ?? null,
                  downloadUrl: item.storageUrl
                    ? await getDownloadUrl(CONSTANTS.BUCKET_DATASETS, item.storageUrl)
                    : null,
                }
              : null,
          }
        }),
      )

      return { errors: rows, total, page, perPage: ERRORS_PER_PAGE }
    },
    {
      runBelongToUser: true,
      query: t.Object({
        page: t.Optional(t.Numeric({ minimum: 1 })),
        classId: t.Optional(t.String({ format: 'uuid' })),
      }),
    },
  )
  /* ── Download the run's training log file via S3 presigned URL (train.py uploads it regardless of whether the live console was ever open) ── */
  .get(
    '/runs/:runId/logs',
    async ({ run }) => {
      const exists = await fileExists(CONSTANTS.BUCKET_TRAINING, trainingLogsKey(run.id))
      if (!exists) return status(404, 'No log file for this run')
      const url = await getDownloadUrl(CONSTANTS.BUCKET_TRAINING, trainingLogsKey(run.id))
      return new Response(null, { status: 302, headers: { Location: url } })
    },
    { runBelongToUser: true },
  )
