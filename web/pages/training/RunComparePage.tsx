import { ArrowLeftIcon, ScalesIcon } from '@phosphor-icons/react'
import { RunComparisonPanel } from '@public/components/training/RunComparisonPanel'
import { EmptyState, LinkButton, PageHeader, QueryBoundary } from '@public/components/ui'
import { projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import { useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/training/compare')

/** Any runs of the project side by side; `?runs=` (comma-separated ids) says which, and the picker edits it. */
export function RunComparePage() {
  const { projectId } = routeApi.useParams()
  const { runs: runIds } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const { data, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs = data?.runs ?? []

  const known = new Set(runs.map((r) => r.id))
  const selectedIds = (runIds ?? '').split(',').filter((id) => id && known.has(id))

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Compare runs"
        description={project.name}
        actions={
          <LinkButton
            to="/project/$projectId/training"
            params={{ projectId }}
            variant="default"
            leftSection={<ArrowLeftIcon size={14} />}
          >
            Back to runs
          </LinkButton>
        }
      />

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {runs.length < 2 ? (
          <EmptyState
            icon={ScalesIcon}
            title="Nothing to compare yet"
            description="Comparing needs at least two runs."
            compact
          />
        ) : (
          <RunComparisonPanel
            projectId={projectId}
            runs={runs}
            selectedIds={selectedIds}
            onSelectedChange={(ids) =>
              void navigate({ search: { runs: ids.length > 0 ? ids.join(',') : undefined }, replace: true })
            }
          />
        )}
      </QueryBoundary>
    </div>
  )
}
