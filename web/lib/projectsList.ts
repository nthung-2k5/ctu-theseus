import type { ProjectSummary } from '@public/lib/api/generated/models'

/**
 * What the projects list does with its data, kept free of React and of the task registry (the caller passes
 * `modalityOf`) so it can be tested without a browser.
 */

export type ProjectSort = 'recent' | 'name' | 'created'

export interface ProjectFilter {
  q: string
  /** A modality id, or null for all of them. */
  modality: string | null
  sort: ProjectSort
}

/** Where a project's one suggested next action leads. `run` carries the id of the run to open. */
export type NextStep =
  | { kind: 'run'; label: string; runId: string }
  | { kind: 'data' | 'snapshot' | 'train' | 'runs'; label: string }

/**
 * The single thing most worth doing in a project, in the order work flows: watch a run that is going, add data,
 * freeze a snapshot, train, then go back through the runs.
 */
export function nextStep(p: ProjectSummary): NextStep {
  if (p.activeRun) return { kind: 'run', label: 'View run', runId: p.activeRun.id }
  if (!p.itemCount) return { kind: 'data', label: 'Add data' }
  if (!p.snapshotCount) return { kind: 'snapshot', label: 'Build a snapshot' }
  if (!p.runCount) return { kind: 'train', label: 'Train' }
  return { kind: 'runs', label: 'View runs' }
}

const time = (iso: string | null | undefined) => (iso ? new Date(iso).getTime() : 0)

/** When the project was last touched: its newest run, else its creation. */
export const activityOf = (p: ProjectSummary) => time(p.lastActivityAt ?? p.createdAt)

export function filterProjects(
  projects: ProjectSummary[],
  { q, modality, sort }: ProjectFilter,
  modalityOf: (task: string) => string,
): ProjectSummary[] {
  const needle = q.trim().toLowerCase()
  return projects
    .filter((p) => !modality || modalityOf(p.task) === modality)
    .filter((p) => !needle || `${p.name} ${p.description ?? ''}`.toLowerCase().includes(needle))
    .sort((a, b) => {
      if (sort === 'name') return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
      if (sort === 'created') return time(b.createdAt) - time(a.createdAt)
      return activityOf(b) - activityOf(a)
    })
}
