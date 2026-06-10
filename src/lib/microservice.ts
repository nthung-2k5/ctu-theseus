/**
 * AI Microservice Integration Layer
 *
 * Communicates with the AI worker service via NATS JetStream.
 *
 * - Model registry is loaded from the AI worker via NATS request-reply at startup
 * - Training/inference/export tasks are published as NATS messages
 * - Results and progress are consumed via NATS durable subscribers
 */

import fs from 'node:fs/promises'
import { db } from '@server/db'
import { trainingMetrics, trainingRuns } from '@server/db/schema'
import { eq } from 'drizzle-orm'
import { t } from 'elysia'
import {
  publishAbortCommand,
  publishExportTask,
  publishInferenceTask,
  publishTrainTask,
  subscribe,
} from './nats'

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

/**
 * Populate the model registry.
 * For now, uses a hardcoded list matching the AI worker's registry.
 * In Phase 3 (plugin system), this will be fetched dynamically from the
 * AI worker via NATS request-reply based on plugin manifests.
 */
export async function refreshModelRegistry(): Promise<void> {
  // Hardcoded registry matching ai_service/models/implementations.py
  // This will be replaced by plugin manifests in Phase 3
  const families: ModelFamilyGroup[] = [
    {
      family: 'resnet',
      variants: [
        { id: 'resnet18', display_name: 'ResNet-18', description: '' },
        { id: 'resnet34', display_name: 'ResNet-34', description: '' },
        { id: 'resnet50', display_name: 'ResNet-50', description: '' },
        { id: 'resnet101', display_name: 'ResNet-101', description: '' },
        { id: 'resnet152', display_name: 'ResNet-152', description: '' },
      ],
    },
    {
      family: 'vit',
      variants: [
        { id: 'vit_tiny', display_name: 'ViT-Tiny', description: '' },
        { id: 'vit_small', display_name: 'ViT-Small', description: '' },
        { id: 'vit_base', display_name: 'ViT-Base', description: '' },
        { id: 'vit_large', display_name: 'ViT-Large', description: '' },
      ],
    },
    {
      family: 'convnext',
      variants: [
        { id: 'convnext_tiny', display_name: 'ConvNeXt-Tiny', description: '' },
        { id: 'convnext_small', display_name: 'ConvNeXt-Small', description: '' },
        { id: 'convnext_base', display_name: 'ConvNeXt-Base', description: '' },
        { id: 'convnext_large', display_name: 'ConvNeXt-Large', description: '' },
      ],
    },
    {
      family: 'mobilenet',
      variants: [
        { id: 'mobilenet_v2', display_name: 'MobileNetV2', description: '' },
        { id: 'mobilenet_v3_small', display_name: 'MobileNetV3-Small', description: '' },
        { id: 'mobilenet_v3_large', display_name: 'MobileNetV3-Large', description: '' },
      ],
    },
    {
      family: 'efficientnet',
      variants: [
        { id: 'efficientnet_b0', display_name: 'EfficientNet-B0', description: '' },
        { id: 'efficientnet_b1', display_name: 'EfficientNet-B1', description: '' },
        { id: 'efficientnet_b2', display_name: 'EfficientNet-B2', description: '' },
        { id: 'efficientnet_b3', display_name: 'EfficientNet-B3', description: '' },
        { id: 'efficientnet_b4', display_name: 'EfficientNet-B4', description: '' },
      ],
    },
  ]

  modelRegistry.clear()
  for (const family of families) {
    for (const variant of family.variants) {
      modelRegistry.set(variant.id, variant)
    }
  }
  console.log(`[microservice] Model registry loaded: ${modelRegistry.size} variants`)
}

// Eagerly populate on import
await refreshModelRegistry()

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
    epochs?: number
  }
}

/**
 * Queue a training job: copy dataset images to ai_mount, then publish task to NATS.
 */
export async function queueTraining(projectId: string, payload: TrainPayload) {
  // Prepare the dataset images
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

  // Publish training task to NATS
  try {
    await publishTrainTask(projectId, {
      id: payload.id,
      type: 'train',
      project_id: projectId,
      payload,
      created_at: new Date().toISOString(),
    })

    // Update status to queued (NATS message ID is the run ID itself)
    await db
      .update(trainingRuns)
      .set({ taskId: payload.id, status: 'queued' })
      .where(eq(trainingRuns.id, payload.id))
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : String(e)
    await db
      .update(trainingRuns)
      .set({ status: 'completed', failedMessage: message, completedAt: new Date() })
      .where(eq(trainingRuns.id, payload.id))
    console.error(`Training dispatch failed: ${e}`)
  }
}

/**
 * Stop a running training task by publishing an abort command to NATS.
 */
export async function stopTraining(runId: string): Promise<{ run_id: string; status: string }> {
  await publishAbortCommand(runId)
  return { run_id: runId, status: 'abort_requested' }
}

/* ------------------------------------------------------------------ */
/*  Inference                                                          */
/* ------------------------------------------------------------------ */

export interface InferencePayload {
  id: string
  model_id: string
  image_path: string
  threshold: number
}

/**
 * Dispatch an inference job via NATS.
 */
export async function dispatchInference(projectId: string, payload: InferencePayload): Promise<void> {
  await publishInferenceTask(projectId, {
    id: payload.id,
    type: 'inference',
    project_id: projectId,
    model_id: payload.model_id,
    image_path: payload.image_path,
    threshold: payload.threshold,
    created_at: new Date().toISOString(),
  })
}

/* ------------------------------------------------------------------ */
/*  Export                                                              */
/* ------------------------------------------------------------------ */

export interface ExportPayload {
  id: string
  model_id: string
  export_format: 'onnx' | 'torchscript'
}

/**
 * Dispatch an export job via NATS.
 */
export async function dispatchExport(projectId: string, payload: ExportPayload): Promise<void> {
  await publishExportTask(projectId, {
    id: payload.id,
    type: 'export',
    project_id: projectId,
    model_id: payload.model_id,
    export_format: payload.export_format,
    created_at: new Date().toISOString(),
  })
}

/* ------------------------------------------------------------------ */
/*  NATS Result/Progress Consumers                                     */
/* ------------------------------------------------------------------ */

/**
 * Start NATS consumers for training results and progress.
 * Called once at gateway startup after initNats().
 */
export async function startNatsConsumers(): Promise<void> {
  // ── Training results ──
  await subscribe('RESULTS', 'theseus.results.train.>', 'gateway-train-results', async (data) => {
    const runId = data.id as string
    const status = data.status as string

    if (!runId) return

    if (status === 'training') {
      // Training has started
      await db
        .update(trainingRuns)
        .set({ status: 'training', startedAt: new Date() })
        .where(eq(trainingRuns.id, runId))
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
        .set({
          status: 'completed',
          failedMessage: (data.error as string) ?? 'Unknown error',
          completedAt: new Date(),
        })
        .where(eq(trainingRuns.id, runId))
    }

    console.log(`[nats] Training result: run=${runId}, status=${status}`)
  })

  // ── Training progress ──
  await subscribe('PROGRESS', 'theseus.progress.train.>', 'gateway-train-progress', async (data) => {
    const runId = data.id as string
    const metrics = data.metrics as Record<string, number> | undefined

    if (!runId || !metrics) return

    await db
      .insert(trainingMetrics)
      .values({
        trainingRunId: runId,
        epoch: metrics.epoch ?? 0,
        trainingLoss: metrics.train_loss ?? 0,
        validationLoss: metrics.val_loss ?? 0,
        accuracy: metrics.accuracy ?? 0,
        mAP: metrics.mAP ?? 0,
      })
      .onConflictDoUpdate({
        target: [trainingMetrics.trainingRunId, trainingMetrics.epoch],
        set: {
          trainingLoss: metrics.train_loss ?? 0,
          validationLoss: metrics.val_loss ?? 0,
          accuracy: metrics.accuracy ?? 0,
          mAP: metrics.mAP ?? 0,
        },
      })

    // Ensure status is 'training'
    await db.update(trainingRuns).set({ status: 'training' }).where(eq(trainingRuns.id, runId))
  })

  // ── Inference results (handled in inference.ts via direct subscription) ──
  // ── Export results (handled in inference.ts via direct subscription) ──

  console.log('[microservice] NATS consumers started for training results and progress.')
}

/* ------------------------------------------------------------------ */
/*  TypeBox schemas (for Elysia route validation)                      */
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
  epochs: t.Optional(t.Integer({ minimum: 1, default: 50 })),
})

export const TrainRequestSchema = t.Object({
  dataset: DatasetConfigSchema,
  model: ModelConfigSchema,
  optimization: t.Optional(OptimizationConfigSchema),
  schedule: t.Optional(ScheduleConfigSchema),
})
