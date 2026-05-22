/**
 * Shared type definitions used across all stores.
 */

import type { Treaty } from '@elysia/eden'
import type { api } from '@public/lib/api'

export type Project = Treaty.Data<typeof api.projects.get>['projects'][number]

export interface DatasetClass {
  id: string
  name: string
  color: string
}

export interface BoundingBox {
  id: string
  imageId: string
  classId: string
  x: number
  y: number
  width: number
  height: number
}

export interface DatasetImage {
  id: string
  filename: string
  url: string
  labels: BoundingBox[]
}

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
