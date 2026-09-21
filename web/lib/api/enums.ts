/** Shared enum unions, taken from the generated models (the OpenAPI document is their only source). */
import type {
  AnnotationOutAnnotationType,
  CreateProjectBodyTask,
  DatasetOutModality,
  DatasetSplitSplitType,
  RunDetailStatus,
  VersionOutStatus,
} from './generated/models'

export type ProjectTask = CreateProjectBodyTask
export type DatasetModality = DatasetOutModality
export type SplitType = DatasetSplitSplitType
export type TrainingStatus = RunDetailStatus
export type DatasetVersionStatus = VersionOutStatus
export type AnnotationType = AnnotationOutAnnotationType
