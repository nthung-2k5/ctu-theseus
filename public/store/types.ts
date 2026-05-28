/**
 * Shared type definitions used across all stores.
 */

import type { Treaty } from '@elysia/eden'
import type { api } from '@public/lib/api'

export type Project = Treaty.Data<ReturnType<typeof api.projects>['get']>['project']

export type DatasetClass = NonNullable<
  Treaty.Data<ReturnType<ReturnType<typeof api.projects>['classes']['get']>>['classes']
>[number]

export interface BoundingBox {
  id: string
  imageId: string
  classId: string
  x: number
  y: number
  width: number
  height: number
}

export type DatasetImage = NonNullable<
  Treaty.Data<ReturnType<ReturnType<typeof api.projects>['images']['get']>>['images'][number]
>

export type DatasetSplitValue = 'train' | 'validation' | 'test'

export interface SplitConfig {
  train: number
  validate: number
  test: number
}

export interface AugmentationConfig {
  id: string
  name: string
  enabled: boolean
  params: Record<string, number | boolean | string>
}

export interface TrainingConfig {
  modelId: string
  variantId: string
  epochs: number
  batchSize: number
  learningRate: number
  imageSize: number
}

export interface TrainingMetrics {
  epoch: number
  trainLoss: number
  valLoss: number
  accuracy: number
  mAP: number
}

export type TrainingStatus = 'idle' | 'queued' | 'running' | 'completed' | 'failed'
