/**
 * Shapes of the gateway's JSON responses, as consumed by the UI.
 *
 * These are hand-maintained rather than derived from Eden: `server/`, `web/`
 * and the repo root are three independent Bun installs (no workspace), so
 * `elysia`'s `Elysia` class resolves to two mutually-unassignable copies and
 * `treaty<App>`-derived types degrade. See the README's "known, deliberate
 * rough edge". The real fix is a Bun workspace with a pinned `elysia`.
 *
 * What is NOT hand-maintained: every enum below is re-exported from
 * `@server/lib/enums`, which plain type imports resolve fine. Only the
 * response *shapes* are written out here.
 */

import type {
  AnnotationType,
  DatasetModality,
  DatasetVersionStatus,
  ProjectTask as ServerProjectTask,
  SplitType as ServerSplitType,
  TrainingStatus as ServerTrainingStatus,
} from '@server/lib/enums'

export type Modality = DatasetModality
export type SplitType = ServerSplitType
export type TrainingStatus = ServerTrainingStatus
export type ProjectTask = ServerProjectTask

/* ── Dataset (1:1 with project, PK = projectId) ── */

export type DatasetSplit = {
  splitType: SplitType
  itemCount?: number
}

export type DatasetVersion = {
  id: string
  datasetId: string
  versionTag: string | null
  status: DatasetVersionStatus
  /** Live membership total, derived server-side from `splitCounts`. */
  itemCount: number | null
  classCount: number | null
  failedMessage: string | null
  parquetKey: string | null
  builtAt: string | Date | null
  createdAt: string | Date
  /** Per-split membership counts, computed server-side by GET /projects/:id. */
  splitCounts?: { train: number; validation: number; test: number }
  /** GET /versions/:id returns computed `splits` instead. */
  splits?: DatasetSplit[]
}

/* ── Label classes (classification tasks only) ── */

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

/* ── Dataset health / EDA ── */

export type ClassCount = {
  classId: string
  name: string
  count: number
}

export type MinMaxAvg = {
  min: number
  max: number
  avg: number
}

export type VisionHealth = {
  count: number
  width: MinMaxAvg
  height: MinMaxAvg
  formats: Record<string, number>
}

export type AudioHealth = {
  count: number
  durationSeconds: MinMaxAvg
  sampleRates: Record<string, number>
}

export type TextHealth = {
  count: number
  tokenCount: MinMaxAvg | null
  languages: Record<string, number>
}

export type TabularHealth = {
  count: number
}

export type DatasetHealthReport = {
  itemCount: number
  labeledCount: number
  unlabeledCount: number
  modality: Modality
  classDistribution: ClassCount[]
  smallClasses: ClassCount[]
  duplicateContentHashes: number
  missingContentHash: number
  vision: VisionHealth | null
  audio: AudioHealth | null
  text: TextHealth | null
  tabular: TabularHealth | null
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

/* ── Annotations ── */

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
  failedMessage: string | null
  startedAt: string | Date | null
  completedAt: string | Date | null
  createdAt: string | Date
  /** Denormalized from the run's evaluation report — see runEvaluations in server/db/schema.ts. Absent/null until the run finishes and evaluates. */
  evaluation?: { status: 'success' | 'failed'; accuracy: number | null; macroF1: number | null } | null
}

/* ── Evaluation ── */
/* Mirrors ai_service/services/evaluate.py's report.json shape (ingested */
/* verbatim into runEvaluations.report — see server/lib/microservice.ts) */
/* and the runEvaluations row GET /runs/:runId/evaluation returns.       */

export type EvaluationStatus = 'success' | 'failed'
export type EvaluationSplitUsed = SplitType | 'full'

export type PerClassStats = {
  precision: number
  recall: number
  f1: number
  support: number
}

export type TopError = {
  itemId: string
  actual: string
  predicted: string
  confidence: number | null
}

export type EvaluationReport = {
  schemaVersion: number
  split: EvaluationSplitUsed
  outputFeature: string
  outputType: 'category' | 'number' | 'sequence' | 'text' | string
  rowCount: number
  /** Classification only — Ludwig's own class-index order. Confusion matrix rows/columns and topErrors' actual/predicted values are always label strings, never bare indices — never re-derive axis labels from the label_classes table, see services/evaluate.py's module docstring. */
  idx2str?: string[]
  truncated?: boolean
  confusionMatrix?: number[][]
  perClass?: Record<string, PerClassStats>
  overall: Record<string, number | null>
  topErrors?: TopError[]
}

export type RunEvaluation = {
  runId: string
  status: EvaluationStatus
  split: EvaluationSplitUsed | null
  reportKey: string | null
  predictionsKey: string | null
  report: EvaluationReport | null
  accuracy: number | null
  macroF1: number | null
  failedMessage: string | null
  evaluatedAt: string | Date
}

export type EvaluationErrorRow = TopError & {
  item: { id: string; text: string | null; downloadUrl: string | null } | null
}

/* ── Sweeps ── */
/* Mirrors server/lib/sweep.ts's SweepSearchSpace and server/routes/sweeps.ts. */

export type SweepStrategyValue = 'grid' | 'random'
export type SweepStatusValue = 'running' | 'completed' | 'canceled'

export type SweepSearchSpace = Partial<{
  epochs: number[]
  batchSize: (number | 'auto')[]
  learningRate: number[]
  earlyStopPatience: number[]
  encoderId: string[]
}>

export type SweepSummary = {
  id: string
  name: string
  strategy: SweepStrategyValue
  maxTrials: number
  status: SweepStatusValue
  createdAt: string | Date
  trialCount: number
  completedTrialCount: number
}

export type SweepTrial = {
  id: string
  name: string
  status: TrainingStatus
  hyperparameters: unknown
  trialIndex: number | null
  failedMessage: string | null
  createdAt: string | Date
  completedAt: string | Date | null
  evaluation?: { status: EvaluationStatus; accuracy: number | null; macroF1: number | null } | null
}

export type SweepDetail = {
  id: string
  projectId: string
  datasetVersionId: string
  name: string
  searchSpace: SweepSearchSpace
  strategy: SweepStrategyValue
  maxTrials: number
  status: SweepStatusValue
  createdAt: string | Date
  updatedAt: string | Date
}
