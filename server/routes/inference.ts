/**
 * Inference routes
 *
 * - POST /api/inference/:runId → Run inference on a trained model
 */

import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { requestInferenceTask, uploadInferenceImage } from '@server/lib/nats'
import { Elysia, status, t } from 'elysia'
import { betterAuth } from './auth'

/* ------------------------------------------------------------------ */
/*  Routes                                                            */
/* ------------------------------------------------------------------ */

export const inferenceRoutes = new Elysia({ prefix: '/api' }).use(betterAuth).post(
  '/inference/:runId',
  async ({ run, body }) => {
    if (run.status !== 'succeeded') {
      return status(404, 'No successfully trained model found')
    }

    const imageBytes = await body.image.bytes()

    const fileExt = path.extname(body.image.name)
    const uploadFilename = `${randomUUID()}${fileExt}`

    // Upload image to NATS Object Store
    const uploadKey = await uploadInferenceImage(uploadFilename, imageBytes)

    // Request via NATS
    const result = await requestInferenceTask(run.id, {
      runId: run.id,
      uploadKey,
      uploadFilename,
      threshold: body.threshold,
    })

    return result
  },
  {
    runBelongToUser: true,
    body: t.Object({
      image: t.File({
        type: ['image/png', 'image/jpeg'],
      }),
      threshold: t.Number(),
    }),
  },
)
