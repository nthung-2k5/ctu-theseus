/**
 * Inference routes
 *
 * - POST /api/inference/:modelId → Run inference on a model
 */

import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { db } from '@server/db'
import { requestInferenceTask, uploadInferenceImage } from '@server/lib/nats'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/* ------------------------------------------------------------------ */
/*  Routes                                                            */
/* ------------------------------------------------------------------ */

export const inferenceRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)
  .post(
    '/inference/:modelId',
    async ({ params, body }) => {
      const run = await db.query.trainingRuns.findFirst({
        where: { id: params.modelId },
        columns: { id: true, failedMessage: true, completedAt: true },
      })

      if (!run || run.failedMessage || !run.completedAt) {
        return status(404, 'No successfully trained model found for this project')
      }

      const imageBytes = await body.image.bytes()

      const fileExt = path.extname(body.image.name)
      const uploadFilename = `${randomUUID()}${fileExt}`

      // Upload image to NATS Object Store
      const uploadKey = await uploadInferenceImage(uploadFilename, imageBytes)

      // Request via NATS
      const result = await requestInferenceTask(run.id, {
        uploadKey,
        threshold: body.threshold,
      })

      return result
    },
    {
      params: t.Object({
        modelId: t.String(),
      }),
      body: t.Object({
        image: t.File({
          type: ['image/png', 'image/jpeg'],
        }),
        threshold: t.Number(),
      }),
    },
  )
