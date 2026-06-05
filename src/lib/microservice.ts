/**
 * AI Microservice Client
 *
 * Typed helpers for communicating with the FastAPI AI service.
 * The model registry is fetched once at startup and cached in-memory.
 */

import fs from 'node:fs/promises'
import { db } from '@server/db'
import { trainingRuns } from '@server/db/schema'
import axios, { AxiosError } from 'axios'
import { eq } from 'drizzle-orm'
import { t } from 'elysia'

export const AI_SERVICE_URL = process.env.AI_SERVICE_URL ?? 'http://localhost:8000'

/* ------------------------------------------------------------------ */
/*  Model Registry                                                     */
/* ------------------------------------------------------------------ */

export interface ModelVariant {
  id: string
  display_name: string
  description: string
}

export interface ModelFamilyGroup {
  family: string
  variants: ModelVariant[]
}

/** Flat lookup map: variant id → variant metadata */
export const modelRegistry = new Map<string, ModelVariant>()

/** Fetch the model registry from the AI service and populate the map. */
export async function refreshModelRegistry(): Promise<void> {
  try {
    const res = await axios.get(`${AI_SERVICE_URL}/api/v1/models`)
    if (res.status !== 200) throw new Error(`HTTP ${res.status}`)
    const families: ModelFamilyGroup[] = res.data
    modelRegistry.clear()
    for (const family of families) {
      for (const variant of family.variants) {
        modelRegistry.set(variant.id, variant)
      }
    }
    console.log(`[microservice] Model registry loaded: ${modelRegistry.size} variants`)
  } catch (error) {
    console.error('[microservice] Failed to fetch model registry:', error)
  }
}

// Eagerly populate on import
await refreshModelRegistry()

/* ------------------------------------------------------------------ */
/*  AI Service API Types                                               */
/* ------------------------------------------------------------------ */

export interface JobResponse {
  job_id: string
  status: string
  message?: string
}

export interface JobStatusResponse {
  job_id: string
  status: string
  result?: Record<string, unknown>
  error?: string
  progress?: {
    epoch: number
    train_loss: number
    val_loss: number
    accuracy: number
    mAP: number
  }
}

/* ------------------------------------------------------------------ */
/*  Training                                                           */
/* ------------------------------------------------------------------ */

export interface TrainPayload {
  id: string
  dataset: {
    source_uri: string
    num_classes: number
    batch_size?: number
    num_workers?: number
  }
  model: {
    architecture: string
    pretrained?: boolean
    drop_rate?: number
  }
  optimization?: {
    optimizer?: 'adamw' | 'adam' | 'sgd'
    learning_rate?: number
    weight_decay?: number
  }
  schedule?: {
    scheduler?: 'cosine' | 'step' | 'linear'
    epochs?: number
    warmup_epochs?: number
  }
  advanced_features?: {
    use_ema?: boolean
    mixup_alpha?: number
    cutmix_alpha?: number
  }
  webhook_url?: string
}

/** Submit a training job to the AI microservice. */
export async function dispatchTraining(payload: TrainPayload): Promise<JobResponse> {
  const res = await axios.post<JobResponse>(`${AI_SERVICE_URL}/api/v1/jobs/train`, payload)
  if (res.status !== 200) {
    throw new Error(`Training dispatch failed (${res.status}): ${res.data}`)
  }
  return res.data
}

/** Stop a running training task. */
export async function stopTraining(taskId: string): Promise<{ job_id: string; status: string }> {
  const res = await axios.delete(`${AI_SERVICE_URL}/api/v1/jobs/${taskId}`)
  if (res.status !== 200) throw new Error(`Stop failed (${res.status})`)
  return res.data
}

/** Get the current status of a Celery task. */
// export async function getTrainingStatus(taskId: string): Promise<JobStatusResponse> {
//   const res = await axios.get<JobStatusResponse>(`${AI_SERVICE_URL}/api/v1/jobs/${taskId}/status`)
//   if (res.status !== 200) throw new Error(`Status fetch failed (${res.status})`)
//   return res.data
// }

/* ------------------------------------------------------------------ */
/*  TypeBox schemas (for Elysia route validation)                     */
/* ------------------------------------------------------------------ */

export const DatasetConfigSchema = t.Object({
  batch_size: t.Optional(t.Integer({ minimum: 1, default: 64 })),
  num_workers: t.Optional(t.Integer({ minimum: 0, default: 4 })),
})

export const ModelConfigSchema = t.Object({
  architecture: t.String({ description: 'Timm model architecture name (registry key)' }),
  pretrained: t.Optional(t.Boolean({ default: true })),
  drop_rate: t.Optional(t.Number({ minimum: 0.0, maximum: 1.0, default: 0.0 })),
})

export const OptimizationConfigSchema = t.Object({
  optimizer: t.Optional(t.Union([t.Literal('adamw'), t.Literal('adam'), t.Literal('sgd')], { default: 'adamw' })),
  learning_rate: t.Optional(t.Number({ exclusiveMinimum: 0.0, default: 0.001 })),
  weight_decay: t.Optional(t.Number({ minimum: 0.0, default: 0.05 })),
})

export const ScheduleConfigSchema = t.Object({
  scheduler: t.Optional(t.Union([t.Literal('cosine'), t.Literal('step'), t.Literal('linear')], { default: 'cosine' })),
  epochs: t.Optional(t.Integer({ minimum: 1, default: 50 })),
  warmup_epochs: t.Optional(t.Integer({ minimum: 0, default: 5 })),
})

export const AdvancedFeaturesConfigSchema = t.Object({
  use_ema: t.Optional(t.Boolean({ default: true })),
  mixup_alpha: t.Optional(t.Number({ minimum: 0.0, default: 0.8 })),
  cutmix_alpha: t.Optional(t.Number({ minimum: 0.0, default: 1.0 })),
})

export const TrainRequestSchema = t.Object({
  dataset: DatasetConfigSchema,
  model: ModelConfigSchema,
  optimization: t.Optional(OptimizationConfigSchema),
  schedule: t.Optional(ScheduleConfigSchema),
  advanced_features: t.Optional(AdvancedFeaturesConfigSchema),
})

export async function queueTraining(projectId: string, payload: TrainPayload) {
  // Prepare the dataset images
  // In this case, we'll copy the images to the ai_mount directory
  const datasetImages = await db.query.datasetImages.findMany({
    where: { projectId, split: { isNotNull: true }, classId: { isNotNull: true } },
    with: {
      datasetClasses: {
        columns: {
          name: true,
        },
      },
    },
  })

  const versionDatasetDir = `./ai_mount/datasets/${payload.id}`

  await Promise.all([
    fs.mkdir(`${versionDatasetDir}/train`, { recursive: true }),
    fs.mkdir(`${versionDatasetDir}/validation`, { recursive: true }),
  ])

  await Promise.all(
    datasetImages
      .filter((datasetImage) => datasetImage.split === 'train' || datasetImage.split === 'validation')
      .map((datasetImage) =>
        Bun.write(
          `${versionDatasetDir}/${datasetImage.split}/${datasetImage.datasetClasses!.name}/${datasetImage.filename}`,
          Bun.file(datasetImage.path),
        ),
      ),
  )

  // Dispatch to the AI microservice
  try {
    const result = await dispatchTraining(payload)

    // Store the Celery task ID
    await db
      .update(trainingRuns)
      .set({ taskId: result.job_id, status: 'queued' })
      .where(eq(trainingRuns.id, payload.id))
  } catch (e: unknown) {
    if (e instanceof AxiosError) {
      await db
        .update(trainingRuns)
        .set({ status: 'completed', failedMessage: e.message, completedAt: new Date() })
        .where(eq(trainingRuns.id, payload.id))
      console.error(`Training dispatch failed: ${e}`)
    }
  }
}

// If either the microservice or backend goes down, we need to update training tasks that are in progress to failed
export const failAllTrainingTasks = async (microserviceSide: boolean) => {
  await db
    .update(trainingRuns)
    .set({
      status: 'completed',
      failedMessage: microserviceSide ? 'AI Service was offline.' : 'Backend was offline.',
    })
    .where(eq(trainingRuns.status, 'training'))
}
