/**
 * Shapes of the API's JSON responses, as consumed by the UI.
 *
 * Almost everything here is a re-export of the Orval-generated models (`lib/api/generated/models`, produced
 * from `schema/openapi.json`) under the names the UI already uses. The exceptions are the evaluation report
 * types: the report is an opaque JSON blob in OpenAPI (it is Ludwig-evaluator output ingested verbatim), so its
 * shape is described here.
 */

import type {
  DatasetModality,
  DatasetVersionStatus,
  SplitType as Split,
  TrainingStatus as Status,
  ProjectTask as Task,
} from '../lib/api/enums'
import type {
  AnnotationOut,
  ProjectDetail as ApiProjectDetail,
  SweepSummary as ApiSweepSummary,
  SweepTrial as ApiSweepTrial,
  DatasetHealth,
  DatasetOut,
  LabelClassOut,
  ProjectSummary,
  RunSummary,
  SearchSpace,
  SweepDetailResponse,
  VersionOut,
} from './../lib/api/generated/models'

export type Modality = DatasetModality
export type SplitType = Split
export type TrainingStatus = Status
export type ProjectTask = Task
export type { DatasetVersionStatus }

export type DatasetVersion = VersionOut
export type LabelClass = LabelClassOut
export type Dataset = DatasetOut
export type DatasetHealthReport = DatasetHealth
export type Project = ProjectSummary
export type ProjectDetail = ApiProjectDetail
export type Annotation = AnnotationOut
export type TrainingRunSummary = RunSummary
export type SweepSummary = ApiSweepSummary
export type SweepTrial = ApiSweepTrial
export type SweepDetail = SweepDetailResponse['sweep']
export type SweepSearchSpace = SearchSpace
export type SweepStrategyValue = 'grid' | 'random'
export type SweepStatusValue = 'running' | 'completed' | 'canceled'

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
  confidence?: number | null
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
