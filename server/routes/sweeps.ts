/**
 * Sweep routes – hyperparameter sweeps, orchestrated on top of the ordinary
 * training pipeline (see server/lib/sweep.ts's module docstring for why
 * this isn't Ludwig's own Ray-Tune-backed hyperopt). A sweep is a search
 * space expanded into N trials, each an ordinary training_runs row
 * (sweepId + trialIndex) dispatched through queueTraining.
 *
 *   POST /api/projects/:projectId/sweeps    – Expand a search space into trials and dispatch them
 *   GET  /api/projects/:projectId/sweeps    – List a project's sweeps, each with its trial count
 *   GET  /api/sweeps/:sweepId               – Sweep detail: every trial + its accuracy/macroF1, reconciled status
 *   POST /api/sweeps/:sweepId/cancel        – Cancel a sweep and every trial that hasn't finished yet
 */

import { db } from '@server/db'
import { SweepStrategies } from '@server/lib/enums'
import { cancelSweep, queueSweep, reconcileSweepStatus } from '@server/lib/microservice'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/** Mirrors server/lib/sweep.ts's SweepSearchSpace — one candidate-value list per sweepable trainer knob. */
const searchSpaceSchema = t.Object({
  epochs: t.Optional(t.Array(t.Number({ minimum: 1 }), { minItems: 1 })),
  batchSize: t.Optional(t.Array(t.Union([t.Number({ minimum: 1 }), t.Literal('auto')]), { minItems: 1 })),
  learningRate: t.Optional(t.Array(t.Number({ minimum: 0, exclusiveMinimum: 0 }), { minItems: 1 })),
  earlyStopPatience: t.Optional(t.Array(t.Number(), { minItems: 1 })),
  encoderId: t.Optional(t.Array(t.String(), { minItems: 1 })),
})

export const sweepRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ── Expand a search space into trials and dispatch them ── */
  .post(
    '/projects/:projectId/sweeps',
    async ({ params, body, project }) => {
      const version = await db.query.datasetVersions.findFirst({
        where: { id: body.datasetVersionId },
        with: { dataset: { columns: { projectId: true } } },
      })
      if (!version) return status(404, 'Dataset version not found')
      if (version.dataset.projectId !== params.projectId) {
        return status(400, 'Dataset version does not belong to this project')
      }

      const result = await queueSweep({
        projectId: params.projectId,
        name: body.name,
        task: project.task,
        datasetVersionId: body.datasetVersionId,
        searchSpace: body.searchSpace,
        strategy: body.strategy,
        maxTrials: body.maxTrials,
      })

      if (!result.ok) return status(result.code, result.message)
      return status(201, { sweep: result.sweep, trials: result.trials })
    },
    {
      projectBelongToUser: true,
      body: t.Object({
        name: t.String({ minLength: 1, maxLength: 255 }),
        datasetVersionId: t.String({ format: 'uuid' }),
        searchSpace: searchSpaceSchema,
        strategy: t.UnionEnum(SweepStrategies),
        maxTrials: t.Integer({ minimum: 1, maximum: 50 }),
      }),
    },
  )

  /* ── List a project's sweeps ── */
  .get(
    '/projects/:projectId/sweeps',
    async ({ params }) => {
      const projectSweeps = await db.query.sweeps.findMany({
        where: { projectId: params.projectId },
        with: {
          trials: {
            columns: { id: true, status: true },
            with: { evaluation: { columns: { accuracy: true, macroF1: true, status: true } } },
          },
        },
        orderBy: { createdAt: 'desc' },
      })

      return {
        sweeps: projectSweeps.map((sweep) => ({
          id: sweep.id,
          name: sweep.name,
          strategy: sweep.strategy,
          maxTrials: sweep.maxTrials,
          status: sweep.status,
          createdAt: sweep.createdAt,
          trialCount: sweep.trials.length,
          completedTrialCount: sweep.trials.filter((t) => t.status !== 'queued' && t.status !== 'running').length,
        })),
      }
    },
    { projectBelongToUser: true },
  )

  /* ── Sweep detail: every trial + its accuracy/macroF1 ── */
  .get(
    '/sweeps/:sweepId',
    async ({ sweep }) => {
      const reconciled = await reconcileSweepStatus(sweep)
      const trials = await db.query.trainingRuns.findMany({
        where: { sweepId: reconciled.id },
        columns: {
          id: true,
          name: true,
          status: true,
          hyperparameters: true,
          trialIndex: true,
          failedMessage: true,
          createdAt: true,
          completedAt: true,
        },
        with: { evaluation: { columns: { status: true, accuracy: true, macroF1: true } } },
        orderBy: { trialIndex: 'asc' },
      })

      return { sweep: reconciled, trials }
    },
    { sweepBelongToUser: true },
  )

  /* ── Cancel a sweep and every trial that hasn't finished yet ── */
  .post(
    '/sweeps/:sweepId/cancel',
    async ({ sweep }) => {
      if (sweep.status !== 'running') {
        return status(409, `Cannot cancel a sweep with status '${sweep.status}' — it has already finished`)
      }
      await cancelSweep(sweep.id)
      return status(204)
    },
    { sweepBelongToUser: true },
  )
