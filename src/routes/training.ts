/**
 * Training routes – connects the ElysiaJS gateway to the AI worker via NATS.
 */

import { db } from '@server/db'
import { datasetClasses, trainingRuns } from '@server/db/schema'
import {
  modelRegistry,
  queueTraining,
  stopTraining,
  type TrainPayload,
  TrainRequestSchema,
} from '@server/lib/microservice'
import { count, eq } from 'drizzle-orm'
import { Elysia, status } from 'elysia'
import { betterAuth } from './auth'

/* ------------------------------------------------------------------ */
/*  Routes                                                            */
/* ------------------------------------------------------------------ */

export const trainingRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  /* ── List training runs for a project ───────────────────────── */
  .get(
    '/projects/:projectId/runs',
    async ({ params }) => {
      const runs = await db.query.trainingRuns.findMany({
        where: {
          projectId: params.projectId,
        },
        columns: {
          id: true,
          taskId: true,
          modelId: true,
          status: true,
          failedMessage: true,
          completedAt: true,
          createdAt: true,
        },
        orderBy: { createdAt: 'desc' },
      })

      return {
        runs: runs.map((run) => {
          let status: "preparing" | "queued" | "training" | "completed" | "failed" | "stopped" = run.status;
          if (run.status === 'completed') {
            if (run.failedMessage) {
              status = "failed"
            }
            else if (run.completedAt === null) {
              status = "stopped"
            }
          }

          return {
            id: run.id,
            taskId: run.taskId,
            modelName: modelRegistry.get(run.modelId)?.display_name ?? run.modelId,
            status,
            createdAt: run.createdAt,
          }
        }),
      }
    },
    { projectBelongToUser: true },
  )
  /* ── Get a single training run with its metrics ─────────────── */
  .get(
    '/runs/:runId',
    async ({ params, user }) => {
      const run = await db.query.trainingRuns.findFirst({
        where: {
          id: params.runId,
          projects: {
            userId: user.id,
          }
        },
        columns: {
          id: false,
          epochs: true,
          failedMessage: true,
          status: true,
          learningRate: false,
          batchSize: false,
          modelId: true,
          taskId: false,
          projectId: false,
          createdAt: true,
          updatedAt: false,
          completedAt: true,
        },
        with: {
          trainingMetrics: true,
        },
      })
      if (!run) return status(404, 'Training run not found')

      let runStatus: "preparing" | "queued" | "training" | "completed" | "failed" | "stopped" = run.status;
      if (run.status === 'completed') {
        if (run.failedMessage) {
          runStatus = "failed"
        }
        else if (run.completedAt === null) {
          runStatus = "stopped"
        }
      }
      return {
        run: {
          ...run,
          status: runStatus,
          modelName: modelRegistry.get(run.modelId)?.display_name ?? run.modelId,
        },
      }
    },
    { auth: true },
  )
  /* ── Start a new training job ──────────────────────────────── */
  .post(
    '/projects/:projectId/train',
    async ({ params, body, project }) => {
      // Validate the model architecture is in the registry
      if (!modelRegistry.has(body.model.architecture)) {
        throw new Error(
          `Unknown model architecture '${body.model.architecture}'. ` +
            `Available: ${[...modelRegistry.keys()].join(', ')}`,
        )
      }

      // Persist the training run in the database
      const [run] = await db
        .insert(trainingRuns)
        .values({
          projectId: params.projectId,
          modelId: body.model.architecture,
          epochs: body.schedule?.epochs ?? 50,
          batchSize: body.dataset.batch_size ?? 64,
          learningRate: body.optimization?.learning_rate ?? 0.001,
          status: 'preparing',
          startedAt: new Date(),
        })
        .returning()

      const payload: TrainPayload = {
        ...body,
        id: run.id,
        dataset: {
          ...body.dataset,
          num_classes: (
            await db
              .select({ count: count() })
              .from(datasetClasses)
              .where(eq(datasetClasses.projectId, params.projectId))
          )[0].count,
        },
      }

      queueTraining(project.id, payload)
    },
    {
      projectBelongToUser: true,
      body: TrainRequestSchema,
    },
  )
  /* ── Stop a training job ───────────────────────────────────── */
  .post('/runs/:runId/stop', async ({ params }) => {
    const run = await db.query.trainingRuns.findFirst({
      where: { id: params.runId },
      columns: { id: true, status: true },
    })
    if (!run) throw new Error('Training run not found')

    // Publish abort command via NATS
    const result = await stopTraining(run.id)

    await db
      .update(trainingRuns)
      // completedAt is null means the training was stopped manually
      // completedAt is not null means the training was completed normally
      .set({ status: 'completed' })
      .where(eq(trainingRuns.id, params.runId))

    return result
  })
  /* ── Get training status ───────────────────────────────────── */
  .get('/runs/:runId/status', async ({ params }) => {
    const run = await db.query.trainingRuns.findFirst({
      where: { id: params.runId },
      columns: { taskId: true, status: true, epochs: true },
    })
    if (!run) return status(404, 'Training run not found')

    const latestMetrics = await db.query.trainingMetrics.findFirst({
      where: { trainingRunId: params.runId },
      columns: { createdAt: false, trainingRunId: false },
      orderBy: {
        createdAt: 'desc'
      }
    })

    const runStatus = {
      status: run.status,
      epochsTotal: run.epochs,
      latestMetrics,
    }

    return runStatus
  })
