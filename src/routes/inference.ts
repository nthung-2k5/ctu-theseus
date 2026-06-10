/**
 * Inference & Export routes
 *
 * - WS   /api/inference/:projectId/ws       → WebSocket: client sends inference/export requests, receives results in real-time
 * - GET  /api/inference/:projectId/download/:format → Download exported model file
 */

import { randomUUID } from 'node:crypto'
import path from 'node:path'
import { db } from '@server/db'
import { trainingRuns } from '@server/db/schema'
import {
  dispatchExport,
  dispatchInference,
} from '@server/lib/microservice'
import { subscribe } from '@server/lib/nats'
import { eq } from 'drizzle-orm'
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
/*  NATS consumers for inference/export results                        */
/* ------------------------------------------------------------------ */

/**
 * Start NATS consumers for inference and export results.
 * Called once at gateway startup after initNats().
 */
export async function startInferenceNatsConsumers(): Promise<void> {
  // ── Inference results ──
  await subscribe('RESULTS', 'theseus.results.inference.>', 'gateway-inference-results', async (data) => {
    const jobId = data.id as string
    const status = data.status as string

    const job = pendingInferenceJobs.get(jobId)
    if (!job) return

    job.status = status === 'success' ? 'success' : 'failed'
    job.result = data.result
    job.error = data.error as string | undefined

    // Push result to all subscribed WebSockets
    for (const wsId of job.subscribers) {
      const ws = activeWebSockets.get(wsId)
      if (ws) {
        ws.send(
          JSON.stringify({
            type: 'inference_result',
            jobId,
            status: job.status,
            result: job.result,
            error: job.error,
          }),
        )
      }
    }

    // Clean up after delivery
    pendingInferenceJobs.delete(jobId)
    console.log(`[nats] Inference result delivered: job=${jobId}, status=${status}`)
  })

  // ── Export results ──
  await subscribe('RESULTS', 'theseus.results.export.>', 'gateway-export-results', async (data) => {
    const jobId = data.id as string
    const status = data.status as string

    const job = pendingExportJobs.get(jobId)
    if (!job) return

    job.status = status === 'success' ? 'success' : 'failed'
    job.result = data.result
    job.error = data.error as string | undefined

    // Push result to all subscribed WebSockets
    for (const wsId of job.subscribers) {
      const ws = activeWebSockets.get(wsId)
      if (ws) {
        ws.send(
          JSON.stringify({
            type: 'export_result',
            jobId,
            status: job.status,
            result: job.result,
            error: job.error,
          }),
        )
      }
    }

    pendingExportJobs.delete(jobId)
    console.log(`[nats] Export result delivered: job=${jobId}, status=${status}`)
  })

  console.log('[inference] NATS consumers started for inference and export results.')
}

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
          const run = await db.query.trainingRuns.findFirst({
            where: { projectId, status: 'completed' },
            columns: { id: true, modelId: true, failedMessage: true, completedAt: true },
            orderBy: { createdAt: 'desc' },
          })

          if (!run || run.failedMessage || !run.completedAt) {
            ws.send(JSON.stringify({ type: 'error', message: 'No successfully trained model found for this project' }))
            return
          }

          // Decode base64 image and save to disk for the AI worker
          const imageData = msg.image as string
          const imageName = (msg.imageName as string) || 'image.jpg'
          const threshold = (msg.threshold as number) ?? 0.5

          const byteString = atob(imageData.split(',')[1] || imageData)
          const ab = new ArrayBuffer(byteString.length)
          const ia = new Uint8Array(ab)
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i)
          }

          // Save to ai_mount/uploads for the worker to read
          const fileExt = path.extname(imageName) || '.jpg'
          const uploadFilename = `${randomUUID()}${fileExt}`
          const uploadPath = `./ai_mount/uploads/${uploadFilename}`
          await Bun.write(uploadPath, new Uint8Array(ab))

          const jobId = randomUUID()

          // Track the pending job
          const jobEntry: PendingJob = {
            projectId,
            subscribers: new Set([wsId]),
            status: 'pending',
          }
          pendingInferenceJobs.set(jobId, jobEntry)

          // Dispatch via NATS
          await dispatchInference(projectId, {
            id: jobId,
            model_id: run.id,
            image_path: `/app_data/uploads/${uploadFilename}`,
            threshold,
          })

          ws.send(JSON.stringify({ type: 'inference_queued', jobId }))
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

          const jobId = randomUUID()

          const jobEntry: PendingJob = {
            projectId,
            subscribers: new Set([wsId]),
            status: 'pending',
          }
          pendingExportJobs.set(jobId, jobEntry)

          // Dispatch via NATS
          await dispatchExport(projectId, {
            id: jobId,
            model_id: run.id,
            export_format: exportFormat,
          })

          ws.send(JSON.stringify({ type: 'export_queued', jobId, format: exportFormat }))
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
