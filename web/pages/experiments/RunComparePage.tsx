import { RunComparisonPanel } from '@public/components/training/RunComparisonPanel'
import { useRunContext } from '@public/components/training/RunContext'
import { useTrainingRuns } from '@public/lib/queries'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/experiments/$runId/compare')

/** The current run is always in the comparison; `?with=` (comma-separated ids) adds the others. */
export function RunComparePage() {
  const { projectId, run } = useRunContext()
  const { with: withIds } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const { data } = useTrainingRuns(projectId)
  const runs = data?.runs ?? []

  const others = (withIds ?? '').split(',').filter((id) => id && id !== run.id)
  const selectedIds = [run.id, ...others]

  return (
    <RunComparisonPanel
      runs={runs}
      selectedIds={selectedIds}
      currentRunId={run.id}
      onSelectedChange={(ids) => {
        const rest = ids.filter((id) => id !== run.id)
        void navigate({ search: { with: rest.length > 0 ? rest.join(',') : undefined }, replace: true })
      }}
    />
  )
}
