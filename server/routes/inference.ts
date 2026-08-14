/**
 * Inference routes
 *
 * - POST /api/inference/:runId → Run inference on a trained model
 *
 * Body shape depends on the run's task modality (server/lib/tasks/registry.ts's
 * itemSpec.payload): file-backed tasks (vision/audio) send `file`, text
 * tasks send `text`, tabular tasks send `record` (JSON-encoded, since
 * multipart form fields are all strings).
 */

import { randomUUID } from 'node:crypto'
import path from 'node:path'
import type { InferenceInputPayload } from '@server/lib/nats'
import { requestInferenceTask, uploadInferenceImage } from '@server/lib/nats'
import { getTaskDescriptor } from '@server/lib/tasks'
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
    if (!run.project) return status(404, 'Run not found')

    const itemPayload = getTaskDescriptor(run.project.task).itemSpec.payload

    let payload: InferenceInputPayload
    if (itemPayload === 'file') {
      if (!body.file) return status(422, 'This task requires a `file` field')
      const fileBytes = await body.file.bytes()
      const fileExt = path.extname(body.file.name)
      const uploadFilename = `${randomUUID()}${fileExt}`
      const uploadKey = await uploadInferenceImage(uploadFilename, fileBytes)
      payload = { kind: 'file', uploadKey, uploadFilename }
    } else if (itemPayload === 'inline_text') {
      if (!body.text) return status(422, 'This task requires a `text` field')
      payload = { kind: 'text', text: body.text }
    } else {
      if (!body.record) return status(422, 'This task requires a `record` field')
      let record: Record<string, string | number>
      try {
        record = JSON.parse(body.record)
      } catch {
        return status(422, '`record` must be JSON-encoded')
      }
      payload = { kind: 'record', record }
    }

    const result = await requestInferenceTask(run.id, {
      runId: run.id,
      threshold: body.threshold,
      payload,
    })

    return result
  },
  {
    runBelongToUser: true,
    body: t.Object({
      file: t.Optional(t.File()),
      text: t.Optional(t.String()),
      /** JSON-encoded record — multipart fields can't carry nested objects. */
      record: t.Optional(t.String()),
      threshold: t.Number({ minimum: 0, maximum: 1, default: 0.5 }),
    }),
  },
)
