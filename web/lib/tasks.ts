/**
 * Read-only view of the backend's task registry.
 *
 * The registry's source of truth is `ai_service/theseus/services/task_registry.py`; the backend dumps it
 * to `schema/task_registry.json` (drift-checked in CI), with the inference input/output kinds already
 * evaluated. This file only types that JSON and re-exposes the helpers the UI used to import from the
 * gateway.
 *
 * This registry is framework-neutral: it describes a task's data shape (item payload, ground truth,
 * snapshot columns, inference contract), never how it trains. Which trainer backends can train a task,
 * and their models/hyperparameters, come from `GET /training-backends` at runtime instead — see
 * `useListProjectTrainingBackends` and `components/training/ParamFields.tsx`.
 */

import registryJson from '@schema/task_registry.json'
import type { AnnotationType, DatasetModality, ProjectTask } from './api/enums'

export type { ProjectTask }

export interface ColumnSpec {
  name: string
  kind: 'storage_uri' | 'inline_text' | 'label' | 'text_sequence_label' | 'split' | 'split_index' | 'scalar' | 'item_id'
}

export type InferenceInputSpec =
  | { kind: 'file'; accept?: string[] }
  | { kind: 'text'; fields: string[] }
  | { kind: 'record' }

export type InferenceOutputKind = 'classification' | 'regression' | 'text' | 'tokens'

export interface TaskOutput {
  column: string
  kind: InferenceOutputKind
}

export interface TaskDescriptor {
  id: ProjectTask
  label: string
  modality: DatasetModality
  status: 'stable' | 'experimental' | 'planned'
  itemSpec: { payload: 'file' | 'inline_text' | 'record'; accept?: string[] }
  annotation: { type: AnnotationType; requiresLabelClasses: boolean }
  columns: ColumnSpec[]
  inputFields: string[]
  /** Absent only for a "planned" task: nothing (registry or backend) knows its output shape yet. */
  output?: TaskOutput
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

/**
 * Tasks Theseus can represent at all (has an annotation/columns story). Whether one is trainable
 * *right now* additionally depends on which trainer backend plugins are installed and available —
 * see `useListTrainingBackends` for that runtime check.
 */
export function listSelectableTasks(): TaskDescriptor[] {
  return Object.values(taskRegistry).filter((descriptor) => descriptor.status !== 'planned')
}
