import { useQuery } from '@tanstack/react-query'
import { edenOptions } from './api'

export const projectsQueryOptions = () => edenOptions.api.projects.get.queryOptions()

export const projectDetailQueryOptions = (projectId: string) =>
  edenOptions.api.projects({ projectId }).get.queryOptions()

export const labelClassesQueryOptions = (projectId: string) =>
  edenOptions.api.projects({ projectId }).classes.get.queryOptions()

export const projectItemsQueryOptions = (
  projectId: string,
  query?: { versionId?: string; split?: 'train' | 'validation' | 'test'; page?: number; perPage?: number },
) => edenOptions.api.projects({ projectId }).items.get.queryOptions(query)

/**
 * Polls every 3s while any run is queued/running — cheap, and it's what
 * catches a run moving out of `queued`. Once a run is active its live
 * status/metrics come from the SSE stream (useRunEvents) instead.
 */
export const trainingRunsQueryOptions = (projectId: string) => ({
  ...edenOptions.api.projects({ projectId }).runs.get.queryOptions(),
  refetchInterval: (query: { state: { data: unknown } }) => {
    const data = query.state.data as { runs: { status: string }[] } | undefined
    if (!data) return false
    const hasActive = data.runs.some((r) => r.status === 'queued' || r.status === 'running')
    return hasActive ? 3000 : false
  },
})

export const trainingRunDetailQueryOptions = (runId: string) => edenOptions.api.runs({ runId }).get.queryOptions()

export function useProjects() {
  return useQuery(projectsQueryOptions())
}

export function useProjectDetail(projectId: string | undefined) {
  return useQuery({ ...projectDetailQueryOptions(projectId ?? ''), enabled: !!projectId })
}

export function useLabelClasses(projectId: string | undefined) {
  return useQuery({ ...labelClassesQueryOptions(projectId ?? ''), enabled: !!projectId })
}

export function useProjectItems(
  projectId: string | undefined,
  query?: { versionId?: string; split?: 'train' | 'validation' | 'test'; page?: number; perPage?: number },
) {
  return useQuery({ ...projectItemsQueryOptions(projectId ?? '', query), enabled: !!projectId })
}

export function useTrainingRuns(projectId: string) {
  return useQuery(trainingRunsQueryOptions(projectId))
}

export function useTrainingRunDetail(runId: string | undefined, enabled: boolean) {
  return useQuery({ ...trainingRunDetailQueryOptions(runId ?? ''), enabled: !!runId && enabled })
}
