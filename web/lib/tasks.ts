/**
 * Read-only view of the backend's task registry.
 *
 * The registry's source of truth is `ai_service/theseus/services/task_registry.py`; the backend dumps it
 * to `schema/task_registry.json` (drift-checked in CI), with the Ludwig feature builders and the inference
 * input/output kinds already evaluated. This file only types that JSON and re-exposes the helpers the UI used
 * to import from the gateway.
 */

import registryJson from '@schema/task_registry.json'
import type { AnnotationType, DatasetModality, ProjectTask } from './api/enums'

export type { ProjectTask }

export interface ColumnSpec {
  name: string
  kind: 'storage_uri' | 'inline_text' | 'label' | 'text_sequence_label' | 'split' | 'split_index' | 'scalar' | 'item_id'
}

export interface LudwigFeature {
  name: string
  type: string
  column: string
  [key: string]: unknown
}

export interface EncoderChoice {
  id: string
  label: string
  encoderType: string
  pretrained: boolean
  params?: Record<string, unknown>
}

export interface TrainerKnobSpec {
  epochs: { default: number; min: number; max: number }
  batchSize: { default: number | 'auto'; options: (number | 'auto')[] }
  learningRate: { default: number; min: number; max: number }
  earlyStopPatience: { default: number; min: number }
}

export type InferenceInputSpec =
  | { kind: 'file'; accept?: string[] }
  | { kind: 'text'; fields: string[] }
  | { kind: 'record' }

export type InferenceOutputKind = 'classification' | 'regression' | 'text' | 'tokens'

export interface TaskDescriptor {
  id: ProjectTask
  label: string
  modality: DatasetModality
  backend: 'ludwig' | 'unsupported'
  status: 'stable' | 'experimental' | 'planned'
  itemSpec: { payload: 'file' | 'inline_text' | 'record'; accept?: string[] }
  annotation: { type: AnnotationType; requiresLabelClasses: boolean }
  columns: ColumnSpec[]
  ludwig?: {
    modelType: 'ecd' | 'llm'
    inputFeatures: LudwigFeature[]
    outputFeatures: LudwigFeature[]
    encoders: EncoderChoice[]
    trainerKnobs: TrainerKnobSpec
  }
  inferenceInputSpec?: InferenceInputSpec
  inferenceOutputKind?: InferenceOutputKind
}

export const taskRegistry = registryJson.tasks as unknown as Record<ProjectTask, TaskDescriptor>

export function getTaskDescriptor(task: ProjectTask): TaskDescriptor {
  return taskRegistry[task]
}

export function getInferenceInputSpec(task: ProjectTask): InferenceInputSpec {
  const spec = getTaskDescriptor(task).inferenceInputSpec
  if (!spec) throw new Error(`Task '${task}' has no inference input spec`)
  return spec
}

export function getInferenceOutputKind(task: ProjectTask): InferenceOutputKind {
  const kind = getTaskDescriptor(task).inferenceOutputKind
  if (!kind) throw new Error(`Task '${task}' has no inference output kind`)
  return kind
}

export function isClassificationTask(task: ProjectTask | undefined): boolean {
  return !!task && getTaskDescriptor(task).annotation.requiresLabelClasses
}

/** Tasks offered on project creation: only what the current backend actually trains. */
export function listSelectableTasks(): TaskDescriptor[] {
  return Object.values(taskRegistry).filter((descriptor) => descriptor.backend === 'ludwig')
}
