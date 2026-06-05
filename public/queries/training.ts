import { createQueryKeys } from '@lukemorales/query-key-factory'
import { wrapEdenFn } from '@public/lib/eden-query'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'

export const training = createQueryKeys('training', {
  runs: (projectId: string) => ({
    queryKey: [projectId],
    queryFn: wrapEdenFn(() => api.projects({ projectId }).runs.get()),
  }),
  runDetail: (runId: string) => ({
    queryKey: [runId],
    queryFn: wrapEdenFn(() => api.runs({ runId }).get()),
  }),
  runStatus: (runId: string) => ({
    queryKey: [runId, 'status'],
    queryFn: wrapEdenFn(() => api.runs({ runId }).status.get()),
  }),
})

/**
 * Fetch all training runs for a project.
 * Polls every 5s when any run is not completed.
 */
export function useTrainingRuns(projectId: string) {
  return useQuery({
    ...training.runs(projectId),
    refetchInterval: (query) => {
      const data = query.state.data as { runs: { status: string }[] } | undefined
      if (!data) return false
      const hasActive = data.runs.some(
        (r) => r.status !== 'completed',
      )
      return hasActive ? 3000 : false
    },
  })
}

/**
 * Fetch the full detail (with metrics) for a single run.
 * Only enabled when the run is completed.
 */
export function useTrainingRunDetail(runId: string | undefined, enabled: boolean) {
  return useQuery({
    ...training.runDetail(runId ?? ''),
    enabled: !!runId && enabled,
  })
}

/**
 * Short-poll the current status + latest metrics for an active run.
 * Polls every 2s while the run is still training/queued.
 */
export function useTrainingRunStatus(runId: string | undefined, isActive: boolean) {
  return useQuery({
    ...training.runStatus(runId ?? ''),
    enabled: !!runId && isActive,
    refetchInterval: isActive ? 2000 : false,
  })
}
