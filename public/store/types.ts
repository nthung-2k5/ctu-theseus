/**
 * Shared type definitions used across all stores.
 * Derived from the Elysia API types via Eden treaty.
 */

import type { ProjectTask as ServerProjectTask } from '@server/lib/enums'

/* ── Modality ── */
export type Modality = 'text' | 'vision' | 'audio' | 'tabular'

/* ── Dataset (1:1 with project, PK = projectId) ── */
export type SplitType = 'train' | 'validation' | 'test'

export type DatasetVersionStatus = 'draft' | 'building' | 'ready' | 'failed'

export type DatasetSplit = {
  splitType: SplitType
  itemCount?: number
}

/** A pool item's membership in a version (draft or snapshot), from GET /projects/:id. */
export type DatasetVersionItem = {
  versionId: string
  itemId: string
  splitType: SplitType
}

export type DatasetVersion = {
  id: string
  datasetId: string
  versionTag: string | null
  augmentationConfig: unknown
  status: DatasetVersionStatus
  itemCount: number | null
  classCount: number | null
  failedMessage: string | null
  createdAt: string | Date
  /** Raw pool membership rows. GET /versions/:id returns computed `splits` instead. */
  items?: DatasetVersionItem[]
  splits?: DatasetSplit[]
}

/* ── Label Classes (for classification tasks) ── */
export type LabelClass = {
  classId: string
  datasetId: string
  name: string
  description: string | null
  uiColorHex: string | null
  createdAt: string | Date | null
}

export type Dataset = {
  projectId: string
  modality: Modality
  createdAt: string | Date
  updatedAt: string | Date
  draft: DatasetVersion | null
  versions: DatasetVersion[]
  classes: LabelClass[]
}

/* ── Project ── */
export type Project = {
  id: string
  name: string
  description: string | null
  task: ProjectTask
  createdAt: string | Date
  draftDataset?: { modality: Modality } | null
}

export type ProjectDetail = {
  id: string
  name: string
  description: string | null
  task: ProjectTask
  userId: string
  createdAt: string | Date
  updatedAt: string | Date
  runCount: number
  versionCount: number
  dataset: Dataset | null
}

/* ── Dataset Items ── */
export type DatasetItem = {
  id: string
  datasetId: string
  externalId: string | null
  storageUrl: string | null
  downloadUrl: string | null
  createdAt: string | Date
  textFeatures: unknown
  visionFeatures: unknown
  audioFeatures: unknown
  tabularFeatures: unknown
  annotations: Annotation[]
}

/* ── Annotations ── */
export type AnnotationType =
  | 'classification'
  | 'bounding_box'
  | 'segmentation_mask'
  | 'text_sequence'
  | 'token_tags'
  | 'preference_rank'

export type Annotation = {
  id: string
  itemId: string
  annotatorId: string | null
  annotationType: AnnotationType
  classId: string | null
  labelTextSequence: string | null
  labelStructured: unknown
  confidenceScore: string | null
  createdAt: string | Date | null
}

/* ── Training ── */
export type TrainingRunSummary = {
  id: string
  name: string
  status: TrainingStatus
  datasetVersionId: string
  startedAt: string | Date | null
  createdAt: string | Date
}

export type TrainingRunDetail = {
  id: string
  name: string
  status: TrainingStatus
  hyperparameters: unknown
  failedMessage: string | null
  startedAt: string | Date | null
  completedAt: string | Date | null
  createdAt: string | Date
  datasetVersion: {
    dataset?: { projectId?: string; modality?: Modality }
  } | null
  metrics: TrainingMetric[]
}

export type TrainingMetric = {
  trainingRunId: string
  epoch: number
  split: SplitType
  metricName: string
  metricValue: number
  createdAt: string | Date
}

export type TrainingStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled'

/**
 * The task registry (`@server/lib/tasks`) is the single source of truth for
 * task→modality/label-classes/trainer-knob behavior; `isClassificationTask`
 * lives there now. This re-export just points ML task IDs at the
 * authoritative enum instead of a second hand-maintained list.
 */
export type ProjectTask = ServerProjectTask
