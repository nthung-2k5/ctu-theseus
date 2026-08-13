import { db } from '@server/db'
import { trainingRuns } from '@server/db/schema'
import { queueTraining, stopTraining } from '@server/lib/microservice'
import { createRunEventsConsumer } from '@server/lib/nats'
import { eq } from 'drizzle-orm'
import { Elysia, sse, status, t } from 'elysia'
import { betterAuth } from './auth'

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
        orderBy: { createdAt: 'desc' },
      })

      return {
        runs: runs.map((run) => ({
          id: run.id,
          name: run.name,
          status: run.status,
          datasetVersionId: run.datasetVersionId,
          startedAt: run.startedAt,
          createdAt: run.createdAt,
        })),
      }
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
        datasetVersionId: t.String(),
        hyperparameters: t.Optional(t.Any()),
      }),
    },
  )
  /* ── Cancel a queued or running training run ── */
  .post(
    '/runs/:runId/cancel',
    async ({ run }) => {
      // Publish abort command via NATS
      await stopTraining(run.id)
      await db.update(trainingRuns).set({ status: 'canceled' }).where(eq(trainingRuns.id, run.id))
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
