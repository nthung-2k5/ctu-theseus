/**
 * Training routes – connects the ElysiaJS gateway to the FastAPI AI microservice.
 */

import assert from 'node:assert'
import { db } from '@server/db'
import { datasetClasses, trainingMetrics, trainingRuns } from '@server/db/schema'
import {
  modelRegistry,
  queueTraining,
  stopTraining,
  type TrainPayload,
  TrainRequestSchema,
} from '@server/lib/microservice'
import { count, eq } from 'drizzle-orm'
import { Elysia, status, t } from 'elysia'
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
          source_uri: `/app_data/datasets/${run.id}`,
          num_classes: (
            await db
              .select({ count: count() })
              .from(datasetClasses)
              .where(eq(datasetClasses.projectId, params.projectId))
          )[0].count,
        },
        webhook_url: '/api/webhooks/training',
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
      columns: { taskId: true },
    })
    if (!run?.taskId) throw new Error('Training run not found or has no task ID')

    const result = await stopTraining(run.taskId)

    await db
      .update(trainingRuns)
      // completedAt is null means the training was stopped manually
      // completedAt is not null means the training was completed normally
      .set({ status: 'completed' })
      .where(eq(trainingRuns.id, params.runId))

    return result
  })
  /* ── Get training status from AI service ───────────────────── */
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
  /* ── Webhook receiver (called by the AI microservice) ──────── */
  .post(
    '/webhooks/training',
    async ({ body }) => {
      const { task_id, status } = body

      // Look up the training run by its Celery task ID
      const run = await db.query.trainingRuns.findFirst({
        where: { taskId: task_id },
        columns: { id: true, status: true },
      })

      if (!run) return

      const runId = run.id

      assert(run.status === 'queued' || run.status === 'training')

      if (status === 'training') {
        const metrics = body.metrics
        if (!metrics) {
          // The task has just started
          await db
            .update(trainingRuns)
            .set({ status: 'training', startedAt: new Date() })
            .where(eq(trainingRuns.id, runId))
        } else {
          // Persist the epoch metrics
          await db
            .insert(trainingMetrics)
            .values({
              trainingRunId: runId,
              epoch: metrics.epoch,
              trainingLoss: metrics.train_loss,
              validationLoss: metrics.val_loss,
              accuracy: metrics.accuracy,
              mAP: metrics.mAP,
            })
            .onConflictDoUpdate({
              target: [trainingMetrics.trainingRunId, trainingMetrics.epoch],
              set: {
                trainingLoss: metrics.train_loss,
                validationLoss: metrics.val_loss,
                accuracy: metrics.accuracy,
                mAP: metrics.mAP,
              },
            })

          // Update training run status (just in case)
          await db.update(trainingRuns).set({ status: 'training' }).where(eq(trainingRuns.id, runId))
        }
      } else if (status === 'completed') {
        await db
          .update(trainingRuns)
          .set({
            status: 'completed',
            completedAt: new Date(),
          })
          .where(eq(trainingRuns.id, runId))
      } else if (status === 'failed') {
        await db
          .update(trainingRuns)
          .set({ failedMessage: body.error, completedAt: new Date() })
          .where(eq(trainingRuns.id, runId))
      }
    },
    {
      body: t.Union([
        t.Object({
          task_id: t.String(),
          status: t.Literal('training'),
          metrics: t.Optional(
            t.Object({
              epoch: t.Number(),
              train_loss: t.Number(),
              val_loss: t.Number(),
              accuracy: t.Number(),
              mAP: t.Number(),
            }),
          ),
        }),
        t.Object({
          task_id: t.String(),
          status: t.Literal('completed'),
          result: t.Object({
            weights: t.String(),
            mapping: t.String(),
          }),
        }),
        t.Object({
          task_id: t.String(),
          status: t.Literal('failed'),
          error: t.String(),
        }),
      ]),
    },
  )
