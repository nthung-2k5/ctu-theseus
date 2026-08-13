import type { DatasetModality, ProjectTask } from '@server/lib/enums'
import { taskRegistry } from './registry'
import type { TaskDescriptor } from './types'

export * from './types'
export { taskRegistry }

export function getTaskDescriptor(task: ProjectTask): TaskDescriptor {
  return taskRegistry[task]
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
