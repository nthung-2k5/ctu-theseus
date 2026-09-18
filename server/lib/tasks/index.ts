import type { DatasetModality, ProjectTask } from '@server/lib/enums'
import { taskRegistry } from './registry'
import type { InferenceInputSpec, InferenceOutputKind, SnapshotContext, TaskDescriptor } from './types'

export * from './types'
export { taskRegistry }

export function getTaskDescriptor(task: ProjectTask): TaskDescriptor {
  return taskRegistry[task]
}

/**
 * What an inference request must supply for this task. For `inline_text`
 * tasks, the field names come from the task's own `ludwig.inputFeatures`
 * builder (called with an empty context — every implemented task's
 * `inputFeatures` ignores its `SnapshotContext` argument, since the columns
 * it names are fixed per task, not dataset-derived) so a multi-input task
 * like question_answering reports both `context` and `question` instead of
 * a single `text` field.
 */
export function getInferenceInputSpec(task: ProjectTask): InferenceInputSpec {
  const descriptor = getTaskDescriptor(task)

  if (descriptor.itemSpec.payload === 'file') {
    return { kind: 'file', accept: descriptor.itemSpec.accept }
  }
  if (descriptor.itemSpec.payload === 'record') {
    return { kind: 'record' }
  }

  if (!descriptor.ludwig) throw new Error(`Task '${task}' has no Ludwig config to derive input fields from`)
  const emptyContext: SnapshotContext = { columns: [], labelClassNames: [] }
  const fields = descriptor.ludwig.inputFeatures(emptyContext).map((f) => f.column)
  return { kind: 'text', fields }
}

/**
 * What shape an inference response takes for this task, derived from the
 * task's Ludwig output feature type (see `build_inference_output` in
 * ai_service/services/predict.py, which this must stay in sync with) — so
 * the UI can pick the right result rendering, and decide whether the
 * confidence-threshold slider applies, before a request ever completes.
 */
export function getInferenceOutputKind(task: ProjectTask): InferenceOutputKind {
  const descriptor = getTaskDescriptor(task)
  if (!descriptor.ludwig) throw new Error(`Task '${task}' has no Ludwig config to derive an output kind from`)
  const emptyContext: SnapshotContext = { columns: [], labelClassNames: [] }
  const outputType = descriptor.ludwig.outputFeatures(emptyContext)[0]?.type

  switch (outputType) {
    case 'category':
      return 'classification'
    case 'number':
      return 'regression'
    case 'sequence':
      return 'tokens'
    default:
      return 'text'
  }
}

export function taskToModality(task: ProjectTask): DatasetModality {
  return getTaskDescriptor(task).modality
}

export function isClassificationTask(task: ProjectTask | undefined): boolean {
  return !!task && getTaskDescriptor(task).annotation.requiresLabelClasses
}

/** Tasks offered on project creation — only what the current backend actually trains. */
export function listSelectableTasks(): TaskDescriptor[] {
  return Object.values(taskRegistry).filter((descriptor) => descriptor.backend === 'ludwig')
}
