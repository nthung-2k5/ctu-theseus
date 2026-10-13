/** Shared enum unions, taken from the generated models (the OpenAPI document is their only source). */
import type {
  AnnotationOutAnnotationType,
  DatasetOutModality,
  DatasetSplitSplitType,
  RunDetailStatus,
  VersionOutStatus,
} from './generated/models'

/** A task id. Tasks are server-side plugins, so the API schema is open-ended; `lib/tasks.ts` has the known ones. */
export type ProjectTask = string
export type DatasetModality = DatasetOutModality
export type SplitType = DatasetSplitSplitType
export type TrainingStatus = RunDetailStatus
export type DatasetVersionStatus = VersionOutStatus
export type AnnotationType = AnnotationOutAnnotationType
