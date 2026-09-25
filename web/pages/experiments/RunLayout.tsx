import { Button, Tabs } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { BrainIcon, ExportIcon, PlayIcon, StopIcon, TrashIcon } from '@phosphor-icons/react'
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

const routeApi = getRouteApi('/_app/project/$projectId/experiments/$runId')

const TABS = ['live', 'logs', 'evaluation', 'compare', 'config'] as const
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

  const goToList = () => navigate({ to: '/project/$projectId/experiments', params: { projectId } })

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
              <LinkButton to="/project/$projectId/experiments" params={{ projectId }} variant="default">
                Back to experiments
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

  const segment = location.pathname.split(`/experiments/${runId}`)[1]?.replace(/^\//, '').split('/')[0] ?? ''
  const activeTab: RunTab = (TABS as readonly string[]).includes(segment) ? (segment as RunTab) : 'live'

  const goToTab = (tab: string | null) => {
    const params = { projectId, runId }
    switch (tab) {
      case 'logs':
        return navigate({ to: '/project/$projectId/experiments/$runId/logs', params })
      case 'evaluation':
        return navigate({ to: '/project/$projectId/experiments/$runId/evaluation', params })
      case 'compare':
        return navigate({ to: '/project/$projectId/experiments/$runId/compare', params, search: {} })
      case 'config':
        return navigate({ to: '/project/$projectId/experiments/$runId/config', params })
      default:
        return navigate({ to: '/project/$projectId/experiments/$runId', params })
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
              {succeeded && (
                <>
                  <LinkButton
                    to="/project/$projectId/playground/$runId"
                    params={{ projectId, runId }}
                    variant="light"
                    leftSection={<PlayIcon size={14} />}
                  >
                    Try
                  </LinkButton>
                  <LinkButton
                    to="/project/$projectId/export/$runId"
                    params={{ projectId, runId }}
                    variant="light"
                    leftSection={<ExportIcon size={14} />}
                  >
                    Export
                  </LinkButton>
                </>
              )}
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
            <Tabs.Tab value="live">Live</Tabs.Tab>
            <Tabs.Tab value="logs">Logs</Tabs.Tab>
            <Tabs.Tab
              value="evaluation"
              disabled={!succeeded}
              title={succeeded ? undefined : 'Available once the run succeeds'}
            >
              Evaluation
            </Tabs.Tab>
            <Tabs.Tab value="compare">Compare</Tabs.Tab>
            <Tabs.Tab value="config">Config</Tabs.Tab>
          </Tabs.List>
        </Tabs>

        <Outlet />
      </div>
    </RunProvider>
  )
}
