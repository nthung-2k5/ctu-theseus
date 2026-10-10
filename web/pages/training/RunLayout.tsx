import { Button, Tabs } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { BrainIcon, ScalesIcon, StopIcon, TrashIcon } from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import { RunProvider } from '@public/components/training/RunContext'
import { confirmDelete, EmptyState, LinkButton, PageHeader, QueryBoundary, StatusBadge } from '@public/components/ui'
import { useRunEvents } from '@public/hooks/useRunEvents'
import {
  cancelRun,
  deleteRun as deleteRunRequest,
  getListRunsQueryKey,
} from '@public/lib/api/generated/training/training'
import { formatDateTime } from '@public/lib/format'
import {
  projectDetailQueryOptions,
  trainingRunDetailQueryOptions,
  trainingRunsQueryOptions,
  useTrainingRuns,
} from '@public/lib/queries'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi, Outlet, useLocation, useNavigate } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/training/$runId')

const TABS = ['live', 'evaluation', 'try', 'export', 'config'] as const
type RunTab = (typeof TABS)[number]

/**
 * Owns the run: one SSE subscription (shared with every tab through context, so it survives tab
 * switches), the header with status and actions, and the URL-driven tab strip.
 */
export function RunLayout() {
  const { projectId, runId } = routeApi.useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const { data: runsData, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const run = runsData?.runs.find((r) => r.id === runId)

  const isActive = run?.status === 'running' || run?.status === 'queued'
  const live = useRunEvents(run?.id, !!isActive, () => {
    queryClient.invalidateQueries({ queryKey: trainingRunsQueryOptions(projectId).queryKey })
    queryClient.invalidateQueries({ queryKey: trainingRunDetailQueryOptions(runId).queryKey })
  })

  const goToList = () => navigate({ to: '/project/$projectId/training', params: { projectId } })

  const stop = useMutation({
    mutationFn: async () => cancelRun(runId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: getListRunsQueryKey(projectId) }),
    onError: () => notifications.show({ title: 'Error', message: 'Failed to stop the run', color: 'red' }),
  })

  const remove = useMutation({
    mutationFn: async () => deleteRunRequest(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getListRunsQueryKey(projectId) })
      void goToList()
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to delete the run', color: 'red' }),
  })

  if (!run) {
    return (
      <div className="flex flex-col gap-3 p-3">
        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          <EmptyState
            icon={BrainIcon}
            title="Run not found"
            description="It may have been deleted."
            action={
              <LinkButton to="/project/$projectId/training" params={{ projectId }} variant="default">
                Back to training
              </LinkButton>
            }
          />
        </QueryBoundary>
      </div>
    )
  }

  const status = live.status ?? run.status
  const succeeded = status === 'succeeded'
  const snapshot = project.dataset?.versions?.find((v) => v.id === run.datasetVersionId)

  const segment = location.pathname.split(`/training/${runId}`)[1]?.replace(/^\//, '').split('/')[0] ?? ''
  const activeTab: RunTab = (TABS as readonly string[]).includes(segment) ? (segment as RunTab) : 'live'

  const goToTab = (tab: string | null) => {
    const params = { projectId, runId }
    switch (tab) {
      case 'evaluation':
        return navigate({ to: '/project/$projectId/training/$runId/evaluation', params })
      case 'try':
        return navigate({ to: '/project/$projectId/training/$runId/try', params })
      case 'export':
        return navigate({ to: '/project/$projectId/training/$runId/export', params })
      case 'config':
        return navigate({ to: '/project/$projectId/training/$runId/config', params })
      default:
        return navigate({ to: '/project/$projectId/training/$runId', params })
    }
  }

  return (
    <RunProvider value={{ projectId, project, run, isActive: !!isActive, status, live }}>
      <div className="flex flex-col gap-3 p-3">
        <PageHeader
          title={run.name}
          badges={<StatusBadge value={status} colorMap={STATUS_COLORS} />}
          description={
            <span className="tnum">
              {run.id.slice(0, 8)} · snapshot {snapshot?.versionTag ?? run.datasetVersionId.slice(0, 8)} · created{' '}
              {formatDateTime(run.createdAt)}
            </span>
          }
          actions={
            <>
              <LinkButton
                to="/project/$projectId/training/compare"
                params={{ projectId }}
                search={{ runs: runId }}
                variant="default"
                leftSection={<ScalesIcon size={14} />}
              >
                Compare
              </LinkButton>
              {isActive && (
                <Button
                  color="red"
                  variant="light"
                  leftSection={<StopIcon weight="fill" size={14} />}
                  loading={stop.isPending}
                  onClick={() => stop.mutate()}
                >
                  Stop
                </Button>
              )}
              <Button
                color="red"
                variant="subtle"
                leftSection={<TrashIcon size={14} />}
                loading={remove.isPending}
                onClick={() =>
                  confirmDelete({
                    title: 'Delete training run',
                    message: (
                      <>
                        Delete <strong>{run.name}</strong>?{' '}
                        {isActive
                          ? 'This stops the in-progress training and permanently removes its metrics and exports.'
                          : 'This permanently removes its metrics and exports.'}{' '}
                        This cannot be undone.
                      </>
                    ),
                    onConfirm: () => remove.mutate(),
                  })
                }
              >
                Delete
              </Button>
            </>
          }
        />

        <Tabs value={activeTab} onChange={goToTab}>
          <Tabs.List>
            <Tabs.Tab value="live">Metrics</Tabs.Tab>
            <Tabs.Tab
              value="evaluation"
              disabled={!succeeded}
              title={succeeded ? undefined : 'Available once the run succeeds'}
            >
              Evaluation
            </Tabs.Tab>
            <Tabs.Tab
              value="try"
              disabled={!succeeded}
              title={succeeded ? undefined : 'Available once the run succeeds'}
            >
              Try
            </Tabs.Tab>
            <Tabs.Tab
              value="export"
              disabled={!succeeded}
              title={succeeded ? undefined : 'Available once the run succeeds'}
            >
              Export
            </Tabs.Tab>
            <Tabs.Tab value="config">Config</Tabs.Tab>
          </Tabs.List>
        </Tabs>

        <Outlet />
      </div>
    </RunProvider>
  )
}
