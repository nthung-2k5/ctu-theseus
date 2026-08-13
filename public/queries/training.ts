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
})

/**
 * Fetch all training runs for a project. Polls every 3s while any run is
 * queued or running — cheap, and it's what catches a run moving out of
 * `queued`. Once a run is active its live status/metrics come from the SSE
 * stream (useRunEvents) instead of polling.
 */
export function useTrainingRuns(projectId: string) {
  return useQuery({
    ...training.runs(projectId),
    refetchInterval: (query) => {
      const data = query.state.data as { runs: { status: string }[] } | undefined
      if (!data) return false
      const hasActive = data.runs.some((r) => r.status === 'queued' || r.status === 'running')
      return hasActive ? 3000 : false
    },
  })
}

/**
 * Fetch the full detail (with metrics) for a single run.
 */
export function useTrainingRunDetail(runId: string | undefined, enabled: boolean) {
  return useQuery({
    ...training.runDetail(runId ?? ''),
    enabled: !!runId && enabled,
  })
}
