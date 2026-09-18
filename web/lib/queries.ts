import type { QueryClient } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { edenOptions } from './api'

export const projectsQueryOptions = () => edenOptions.api.projects.get.queryOptions()

export const projectDetailQueryOptions = (projectId: string) =>
  edenOptions.api.projects({ projectId }).get.queryOptions()

export const labelClassesQueryOptions = (projectId: string) =>
  edenOptions.api.projects({ projectId }).classes.get.queryOptions()

export const datasetHealthQueryOptions = (projectId: string) =>
  edenOptions.api.projects({ projectId }).dataset.health.get.queryOptions()

export const projectItemsQueryOptions = (
  projectId: string,
  query?: {
    versionId?: string
    split?: 'train' | 'validation' | 'test'
    classId?: string
    search?: string
    page?: number
    perPage?: number
    sort?: 'newest' | 'oldest' | 'filename'
  },
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

/** Polls while the run is still active — no evaluation row exists until the worker finishes training. */
export const runEvaluationQueryOptions = (runId: string, isActive: boolean) => ({
  ...edenOptions.api.runs({ runId }).evaluation.get.queryOptions(),
  retry: false,
  refetchInterval: (isActive ? 5000 : false) as number | false,
})

export const runEvaluationErrorsQueryOptions = (runId: string, page: number, classId?: string) =>
  edenOptions.api.runs({ runId }).evaluation.errors.get.queryOptions({ page, ...(classId && { classId }) })

/** Polls while any sweep still has a running/queued trial. */
export const projectSweepsQueryOptions = (projectId: string) => ({
  ...edenOptions.api.projects({ projectId }).sweeps.get.queryOptions(),
  refetchInterval: (query: { state: { data: unknown } }) => {
    const data = query.state.data as { sweeps: { status: string }[] } | undefined
    if (!data) return false
    return data.sweeps.some((s) => s.status === 'running') ? 5000 : false
  },
})

export const sweepDetailQueryOptions = (sweepId: string) => ({
  ...edenOptions.api.sweeps({ sweepId }).get.queryOptions(),
  refetchInterval: (query: { state: { data: unknown } }) => {
    const data = query.state.data as { sweep: { status: string } } | undefined
    return data?.sweep.status === 'running' ? 5000 : false
  },
})

export function useProjects() {
  return useQuery(projectsQueryOptions())
}

export function useProjectDetail(projectId: string | undefined) {
  return useQuery({ ...projectDetailQueryOptions(projectId ?? ''), enabled: !!projectId })
}

export function useLabelClasses(projectId: string | undefined) {
  return useQuery({ ...labelClassesQueryOptions(projectId ?? ''), enabled: !!projectId })
}

export function useDatasetHealth(projectId: string | undefined, enabled: boolean) {
  return useQuery({ ...datasetHealthQueryOptions(projectId ?? ''), enabled: !!projectId && enabled })
}

export function useProjectItems(
  projectId: string | undefined,
  query?: {
    versionId?: string
    split?: 'train' | 'validation' | 'test'
    classId?: string
    search?: string
    page?: number
    perPage?: number
    sort?: 'newest' | 'oldest' | 'filename'
  },
) {
  return useQuery({ ...projectItemsQueryOptions(projectId ?? '', query), enabled: !!projectId })
}

export function useTrainingRuns(projectId: string) {
  return useQuery(trainingRunsQueryOptions(projectId))
}

export function useTrainingRunDetail(runId: string | undefined, enabled: boolean) {
  return useQuery({ ...trainingRunDetailQueryOptions(runId ?? ''), enabled: !!runId && enabled })
}

export function useRunEvaluation(runId: string | undefined, isActive: boolean) {
  return useQuery({ ...runEvaluationQueryOptions(runId ?? '', isActive), enabled: !!runId })
}

export function useRunEvaluationErrors(
  runId: string | undefined,
  page: number,
  classId: string | undefined,
  enabled: boolean,
) {
  return useQuery({ ...runEvaluationErrorsQueryOptions(runId ?? '', page, classId), enabled: !!runId && enabled })
}

export function useProjectSweeps(projectId: string) {
  return useQuery(projectSweepsQueryOptions(projectId))
}

export function useSweepDetail(sweepId: string | undefined) {
  return useQuery({ ...sweepDetailQueryOptions(sweepId ?? ''), enabled: !!sweepId })
}

/* ── Invalidation ──────────────────────────────────────────────────────── */

/**
 * Invalidate everything scoped to one project.
 *
 * Project detail is not just "the project row": it carries the draft's item
 * membership (the split-count bar reads it) and the label class list (the
 * sidebar badge reads it). So a mutation that changes items, splits, or
 * classes has to invalidate project detail too, not only the query it most
 * obviously touches. Invalidating each of those independently is how the
 * split counts and the class badge ended up going stale after some mutations
 * but not others.
 */
export function invalidateProjectScope(queryClient: QueryClient, projectId: string) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: edenOptions.api.projects({ projectId }).get.queryKey() }),
    queryClient.invalidateQueries({ queryKey: edenOptions.api.projects({ projectId }).items.get.queryKey() }),
    queryClient.invalidateQueries({ queryKey: edenOptions.api.projects({ projectId }).classes.get.queryKey() }),
  ])
}

/** Invalidate the project list (after create/rename/delete). */
export function invalidateProjectList(queryClient: QueryClient) {
  return queryClient.invalidateQueries({ queryKey: edenOptions.api.projects.get.queryKey() })
}

/** Invalidate a project's training runs (after dispatch/cancel/delete). */
export function invalidateRunScope(queryClient: QueryClient, projectId: string) {
  return queryClient.invalidateQueries({ queryKey: edenOptions.api.projects({ projectId }).runs.get.queryKey() })
}
