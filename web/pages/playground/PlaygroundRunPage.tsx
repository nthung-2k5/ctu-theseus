import { Select } from '@mantine/core'
import { RunInferencePanel } from '@public/components/training/RunInferencePanel'
import { EmptyState, LinkButton, PageHeader, QueryBoundary } from '@public/components/ui'
import { useTrainingRuns } from '@public/lib/queries'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/playground/$runId')

export function PlaygroundRunPage() {
  const { projectId, runId } = routeApi.useParams()
  const navigate = routeApi.useNavigate()
  const { data, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs = data?.runs ?? []
  const run = runs.find((r) => r.id === runId)
  const trained = runs.filter((r) => r.status === 'succeeded')

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title={run ? `Playground · ${run.name}` : 'Playground'}
        description="Provide an input, run the model, and inspect the prediction."
        actions={
          <>
            <Select
              size="xs"
              w={240}
              aria-label="Run"
              data={trained.map((r) => ({ value: r.id, label: r.name }))}
              value={run?.status === 'succeeded' ? run.id : null}
              onChange={(id) =>
                id && navigate({ to: '/project/$projectId/playground/$runId', params: { projectId, runId: id } })
              }
              allowDeselect={false}
              placeholder="Choose a trained run"
            />
            <LinkButton to="/project/$projectId/playground" params={{ projectId }} variant="default">
              All models
            </LinkButton>
          </>
        }
      />

      {run ? (
        <RunInferencePanel projectId={projectId} run={run} />
      ) : (
        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          <EmptyState title="Run not found" description="It may have been deleted." />
        </QueryBoundary>
      )}
    </div>
  )
}
