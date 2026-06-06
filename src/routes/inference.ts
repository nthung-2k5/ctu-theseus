/**
 * Inference & Export routes
 *
 * - POST /api/inference/:projectId/infer    → dispatches an inference job via the AI service
 * - POST /api/inference/:projectId/export   → dispatches an export job via the AI service
 * - POST /api/webhooks/inference            → webhook receiver for inference results
 * - POST /api/webhooks/export               → webhook receiver for export results
 * - WS   /api/inference/:projectId/ws       → WebSocket: client sends inference requests, receives results in real-time
 */

import { db } from '@server/db'
import { datasetClasses, trainingRuns } from '@server/db/schema'
import {
  AI_SERVICE_URL,
  dispatchExport,
  dispatchInference,
  modelRegistry,
  type ExportPayload,
} from '@server/lib/microservice'
import { count, eq } from 'drizzle-orm'
import { Elysia, t } from 'elysia'
import { betterAuth } from './auth'

/* ------------------------------------------------------------------ */
/*  In-memory tracking for pending jobs                                */
/* ------------------------------------------------------------------ */

interface PendingJob {
  projectId: string
  /** Set of WebSocket IDs subscribed to this job */
  subscribers: Set<string>
  result?: unknown
  error?: string
  status: 'pending' | 'success' | 'failed'
}

/** job_id → PendingJob */
const pendingInferenceJobs = new Map<string, PendingJob>()
const pendingExportJobs = new Map<string, PendingJob>()

/** ws id → WebSocket instance (for pushing results) */
const activeWebSockets = new Map<string, { send: (data: string) => void; projectId: string }>()

let wsIdCounter = 0

/* ------------------------------------------------------------------ */
/*  Routes                                                             */
/* ------------------------------------------------------------------ */

export const inferenceRoutes = new Elysia({ prefix: '/api' })
  .use(betterAuth)

  /* ── WebSocket: real-time inference channel ──────────────────── */
  .ws('/inference/:projectId/ws', {
    open(ws) {
      const wsId = `ws_${++wsIdCounter}`
      const projectId = ws.data.params.projectId
      ws.data.wsId = wsId
      activeWebSockets.set(wsId, { send: (d) => ws.send(d), projectId })
      ws.send(JSON.stringify({ type: 'connected', wsId }))
    },

    async message(ws, message) {
      const wsId = ws.data.wsId as string
      const projectId = ws.data.params.projectId

      try {
        const msg = typeof message === 'string' ? JSON.parse(message) : message

        if (msg.type === 'inference') {
          // Client wants to run inference
          // We need the training run to know model_id and model_name
          const run = await db.query.trainingRuns.findFirst({
            where: { projectId, status: 'completed' },
            columns: { id: true, modelId: true, failedMessage: true, completedAt: true },
            orderBy: { createdAt: 'desc' },
          })

          if (!run || run.failedMessage || !run.completedAt) {
            ws.send(JSON.stringify({ type: 'error', message: 'No successfully trained model found for this project' }))
            return
          }

          // Fetch the image from the provided base64 data
          const imageData = msg.image as string // base64-encoded image
          const imageName = (msg.imageName as string) || 'image.jpg'
          const threshold = (msg.threshold as number) ?? 0.5

          // Convert base64 to File
          const byteString = atob(imageData.split(',')[1] || imageData)
          const mimeMatch = imageData.match(/data:([^;]+);/)
          const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg'
          const ab = new ArrayBuffer(byteString.length)
          const ia = new Uint8Array(ab)
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i)
          }
          const imageFile = new File([ab], imageName, { type: mimeType })

          const modelName = run.modelId
          const modelId = run.id

          const result = await dispatchInference({
            model_id: modelId,
            model_name: modelName,
            threshold,
            webhook_url: '/api/webhooks/inference',
            image: imageFile,
          })

          // Track the pending job
          const jobEntry: PendingJob = {
            projectId,
            subscribers: new Set([wsId]),
            status: 'pending',
          }
          pendingInferenceJobs.set(result.job_id, jobEntry)

          ws.send(JSON.stringify({ type: 'inference_queued', jobId: result.job_id }))
        } else if (msg.type === 'export') {
          // Client wants to export a model
          const run = await db.query.trainingRuns.findFirst({
            where: { projectId, status: 'completed' },
            columns: { id: true, modelId: true, failedMessage: true, completedAt: true },
            orderBy: { createdAt: 'desc' },
          })

          if (!run || run.failedMessage || !run.completedAt) {
            ws.send(JSON.stringify({ type: 'error', message: 'No successfully trained model found for this project' }))
            return
          }

          const exportFormat = msg.format as 'onnx' | 'torchscript'
          if (!['onnx', 'torchscript'].includes(exportFormat)) {
            ws.send(JSON.stringify({ type: 'error', message: 'Invalid export format. Use "onnx" or "torchscript".' }))
            return
          }

          // Get num_classes
          const classCount = (
            await db
              .select({ count: count() })
              .from(datasetClasses)
              .where(eq(datasetClasses.projectId, projectId))
          )[0].count

          const result = await dispatchExport({
            model_id: run.id,
            model_name: run.modelId,
            num_classes: classCount,
            export_format: exportFormat,
            webhook_url: '/api/webhooks/export',
          })

          const jobEntry: PendingJob = {
            projectId,
            subscribers: new Set([wsId]),
            status: 'pending',
          }
          pendingExportJobs.set(result.job_id, jobEntry)

          ws.send(JSON.stringify({ type: 'export_queued', jobId: result.job_id, format: exportFormat }))
        }
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err)
        ws.send(JSON.stringify({ type: 'error', message: errMsg }))
      }
    },

    close(ws) {
      const wsId = ws.data.wsId as string
      activeWebSockets.delete(wsId)
    },

    params: t.Object({
      projectId: t.String(),
    }),
    body: t.Any(),
  })

  /* ── Webhook: inference result from AI service ──────────────── */
  .post(
    '/webhooks/inference',
    ({ body }) => {
      const { task_id, status, result, error } = body

      const job = pendingInferenceJobs.get(task_id)
      if (!job) return { received: true }

      job.status = status === 'success' ? 'success' : 'failed'
      job.result = result
      job.error = error

      // Push result to all subscribed WebSockets
      for (const wsId of job.subscribers) {
        const ws = activeWebSockets.get(wsId)
        if (ws) {
          ws.send(
            JSON.stringify({
              type: 'inference_result',
              jobId: task_id,
              status: job.status,
              result: job.result,
              error: job.error,
            }),
          )
        }
      }

      // Clean up after delivery
      pendingInferenceJobs.delete(task_id)
      return { received: true }
    },
    {
      body: t.Object({
        task_id: t.String(),
        status: t.String(),
        result: t.Optional(t.Any()),
        error: t.Optional(t.String()),
      }),
    },
  )

  /* ── Webhook: export result from AI service ─────────────────── */
  .post(
    '/webhooks/export',
    ({ body }) => {
      const { task_id, status, result, error } = body

      const job = pendingExportJobs.get(task_id)
      if (!job) return { received: true }

      job.status = status === 'success' ? 'success' : 'failed'
      job.result = result
      job.error = error

      // Push result to all subscribed WebSockets
      for (const wsId of job.subscribers) {
        const ws = activeWebSockets.get(wsId)
        if (ws) {
          ws.send(
            JSON.stringify({
              type: 'export_result',
              jobId: task_id,
              status: job.status,
              result: job.result,
              error: job.error,
            }),
          )
        }
      }

      pendingExportJobs.delete(task_id)
      return { received: true }
    },
    {
      body: t.Object({
        task_id: t.String(),
        status: t.String(),
        result: t.Optional(t.Any()),
        error: t.Optional(t.String()),
      }),
    },
  )

  /* ── REST: download exported model file ─────────────────────── */
  .get(
    '/inference/:projectId/download/:format',
    async ({ params }) => {
      const run = await db.query.trainingRuns.findFirst({
        where: { projectId: params.projectId, status: 'completed' },
        columns: { id: true, failedMessage: true, completedAt: true },
        orderBy: { createdAt: 'desc' },
      })

      if (!run || run.failedMessage || !run.completedAt) {
        throw new Error('No successfully trained model found')
      }

      const modelDir = `./ai_mount/models`
      const filePath = `${modelDir}/${run.id}.${params.format}`
      const file = Bun.file(filePath)

      if (!(await file.exists())) {
        throw new Error(`Exported file not found. Run export first.`)
      }

      return new Response(file, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Disposition': `attachment; filename="model.${params.format}"`,
        },
      })
    },
    {
      params: t.Object({
        projectId: t.String(),
        format: t.String(),
      }),
    },
  )
