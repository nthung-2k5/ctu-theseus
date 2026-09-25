import {
  Badge,
  Button,
  Checkbox,
  Group,
  Loader,
  MultiSelect,
  SegmentedControl,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  BrainIcon,
  FlaskIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  ScalesIcon,
  TrashIcon,
  TrophyIcon,
} from '@phosphor-icons/react'
import { STATUS_COLORS } from '@public/components/training/constants'
import {
  confirmDelete,
  DataTable,
  type DataTableColumn,
  EmptyState,
  LinkButton,
  PageHeader,
  QueryBoundary,
  SectionLabel,
  StatusBadge,
} from '@public/components/ui'
import { deleteRun as deleteRunRequest, getListRunsQueryKey } from '@public/lib/api/generated/training/training'
import { projectDetailQueryOptions, useProjectSweeps, useTrainingRuns } from '@public/lib/queries'
import type { SweepSummary, TrainingRunSummary } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useMemo, useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/experiments/')

const STATUSES = ['queued', 'running', 'succeeded', 'failed', 'canceled']

const SWEEP_STATUS_COLORS: Record<string, string> = { running: 'cyan', completed: 'teal', canceled: 'gray' }

export function ExperimentsPage() {
  const { projectId } = routeApi.useParams()
  const navigate = routeApi.useNavigate()
  const queryClient = useQueryClient()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))

  const { data: runsData, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs: TrainingRunSummary[] = runsData?.runs ?? []
  const { data: sweepsData } = useProjectSweeps(projectId)
  const sweeps: SweepSummary[] = sweepsData?.sweeps ?? []

  const hasReadySnapshot = project.dataset?.versions?.some((v) => v.status === 'ready') ?? false
  const versionTag = (id: string) => project.dataset?.versions?.find((v) => v.id === id)?.versionTag ?? id.slice(0, 8)

  const [search, setSearch] = useState('')
  const [statuses, setStatuses] = useState<string[]>([])
  const [sort, setSort] = useState<'newest' | 'best'>('newest')
  const [compareIds, setCompareIds] = useState<Set<string>>(new Set())

  const toggleCompare = (runId: string) =>
    setCompareIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })

  // The run with the highest accuracy across every successful evaluation in the project.
  const bestRunId = runs.reduce<{ id: string; accuracy: number } | null>((best, run) => {
    const accuracy = run.evaluation?.status === 'success' ? run.evaluation.accuracy : null
    if (accuracy == null) return best
    return !best || accuracy > best.accuracy ? { id: run.id, accuracy } : best
  }, null)?.id

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase()
    const accuracyOf = (r: TrainingRunSummary) =>
      r.evaluation?.status === 'success' ? (r.evaluation.accuracy ?? -1) : -1
    return runs
      .filter((r) => (q ? r.name.toLowerCase().includes(q) || r.id.startsWith(q) : true))
      .filter((r) => (statuses.length > 0 ? statuses.includes(r.status) : true))
      .sort((a, b) =>
        sort === 'best'
          ? accuracyOf(b) - accuracyOf(a)
          : new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
  }, [runs, search, statuses, sort])

  const deleteRun = useMutation({
    mutationFn: async (runId: string) => deleteRunRequest(runId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: getListRunsQueryKey(projectId) }),
    onError: () => notifications.show({ title: 'Error', message: 'Failed to delete the run', color: 'red' }),
  })

  const handleDelete = (run: TrainingRunSummary) => {
    const isActive = run.status === 'running' || run.status === 'queued'
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
      onConfirm: () => deleteRun.mutate(run.id),
    })
  }

  const openRun = (runId: string) =>
    navigate({ to: '/project/$projectId/experiments/$runId', params: { projectId, runId } })

  const compare = () => {
    const [first, ...rest] = [...compareIds]
    if (!first) return
    void navigate({
      to: '/project/$projectId/experiments/$runId/compare',
      params: { projectId, runId: first },
      search: { with: rest.join(',') || undefined },
    })
  }

  const runColumns: DataTableColumn<TrainingRunSummary>[] = [
    {
      key: 'compare',
      header: '',
      fit: true,
      render: (run) => (
        <Checkbox
          checked={compareIds.has(run.id)}
          onChange={() => toggleCompare(run.id)}
          onClick={(e) => e.stopPropagation()}
          size="xs"
          aria-label={`Compare ${run.name}`}
        />
      ),
    },
    {
      key: 'name',
      header: 'Run',
      render: (run) => (
        <div>
          <Group gap={6} wrap="nowrap">
            <Text size="sm" fw={500}>
              {run.name}
            </Text>
            {run.id === bestRunId && (
              <Tooltip label="Highest accuracy in this project">
                <TrophyIcon size={14} weight="fill" color="var(--mantine-color-yellow-6)" />
              </Tooltip>
            )}
          </Group>
          <Text size="xs" c="dimmed" className="tnum">
            {run.id.slice(0, 8)}
          </Text>
        </div>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (run) => (
        <Group gap={4} wrap="nowrap">
          {(run.status === 'running' || run.status === 'queued') && <Loader size={10} />}
          <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
        </Group>
      ),
    },
    {
      key: 'snapshot',
      header: 'Snapshot',
      fit: true,
      render: (run) => (
        <Text size="xs" c="dimmed">
          {versionTag(run.datasetVersionId)}
        </Text>
      ),
    },
    {
      key: 'accuracy',
      header: 'Accuracy',
      fit: true,
      render: (run) => (
        <Text size="sm" className="tnum" c={run.evaluation?.accuracy != null ? undefined : 'dimmed'}>
          {run.evaluation?.status === 'success' && run.evaluation.accuracy != null
            ? run.evaluation.accuracy.toFixed(3)
            : '—'}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (run) => (
        <Text size="xs" c="dimmed" className="tnum">
          {new Date(run.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (run) => (
        <Group gap={4} wrap="nowrap" onClick={(e) => e.stopPropagation()}>
          <LinkButton
            to="/project/$projectId/experiments/$runId/evaluation"
            params={{ projectId, runId: run.id }}
            size="compact-xs"
            variant="light"
            disabled={run.status !== 'succeeded'}
          >
            Evaluate
          </LinkButton>
          <LinkButton
            to="/project/$projectId/playground/$runId"
            params={{ projectId, runId: run.id }}
            size="compact-xs"
            variant="light"
            disabled={run.status !== 'succeeded'}
          >
            Try
          </LinkButton>
          <Button
            size="compact-xs"
            variant="subtle"
            color="red"
            aria-label={`Delete ${run.name}`}
            onClick={() => handleDelete(run)}
          >
            <TrashIcon size={14} />
          </Button>
        </Group>
      ),
    },
  ]

  const sweepColumns: DataTableColumn<SweepSummary>[] = [
    {
      key: 'name',
      header: 'Sweep',
      render: (sweep) => (
        <Text size="sm" fw={500}>
          {sweep.name}
        </Text>
      ),
    },
    {
      key: 'strategy',
      header: 'Strategy',
      fit: true,
      render: (sweep) => (
        <Text size="sm" tt="capitalize">
          {sweep.strategy}
        </Text>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (sweep) => (
        <Badge color={SWEEP_STATUS_COLORS[sweep.status] ?? 'gray'} tt="capitalize">
          {sweep.status === 'running' && <Loader size={8} mr={4} />}
          {sweep.status}
        </Badge>
      ),
    },
    {
      key: 'trials',
      header: 'Trials',
      fit: true,
      render: (sweep) => (
        <Text size="sm" c="dimmed" className="tnum">
          {sweep.completedTrialCount} / {sweep.trialCount}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (sweep) => (
        <Text size="xs" c="dimmed" className="tnum">
          {new Date(sweep.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Experiments"
        description="Training runs and sweeps: monitor them live, then evaluate, test or export the result."
        actions={
          <>
            {compareIds.size > 0 && (
              <Button
                variant="light"
                leftSection={<ScalesIcon size={14} />}
                disabled={compareIds.size < 2}
                onClick={compare}
              >
                Compare ({compareIds.size})
              </Button>
            )}
            <LinkButton
              to="/project/$projectId/experiments/new"
              params={{ projectId }}
              search={{ mode: 'sweep' }}
              variant="default"
              leftSection={<FlaskIcon size={14} />}
              disabled={!hasReadySnapshot}
            >
              New sweep
            </LinkButton>
            <LinkButton
              to="/project/$projectId/experiments/new"
              params={{ projectId }}
              search={{ mode: 'run' }}
              leftSection={<PlusIcon size={14} />}
              disabled={!hasReadySnapshot}
            >
              New run
            </LinkButton>
          </>
        }
      />

      {runs.length > 0 && (
        <Group gap="xs" wrap="wrap">
          <TextInput
            size="xs"
            placeholder="Search runs…"
            leftSection={<MagnifyingGlassIcon size={14} />}
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            style={{ flexGrow: 1, minWidth: 200, maxWidth: 320 }}
          />
          <MultiSelect
            size="xs"
            placeholder={statuses.length === 0 ? 'All statuses' : undefined}
            data={STATUSES}
            value={statuses}
            onChange={setStatuses}
            clearable
            w={220}
          />
          <SegmentedControl
            ml="auto"
            value={sort}
            onChange={(v) => setSort(v as 'newest' | 'best')}
            data={[
              { value: 'newest', label: 'Newest' },
              { value: 'best', label: 'Best accuracy' },
            ]}
          />
        </Group>
      )}

      <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
        {runs.length === 0 ? (
          <EmptyState
            icon={BrainIcon}
            title="No runs yet"
            description={
              hasReadySnapshot ? 'Create your first training run.' : 'Build a snapshot first, then train on it.'
            }
            action={
              hasReadySnapshot ? (
                <LinkButton
                  to="/project/$projectId/experiments/new"
                  params={{ projectId }}
                  search={{ mode: 'run' }}
                  leftSection={<PlusIcon size={14} />}
                >
                  New run
                </LinkButton>
              ) : (
                <LinkButton to="/project/$projectId/snapshots/new" params={{ projectId }} variant="default">
                  Build a snapshot
                </LinkButton>
              )
            }
          />
        ) : (
          <DataTable
            columns={runColumns}
            data={visible}
            getRowKey={(run) => run.id}
            onRowClick={(run) => openRun(run.id)}
            emptyMessage="No runs match the filters"
          />
        )}
      </QueryBoundary>

      {sweeps.length > 0 && (
        <div className="flex flex-col gap-2">
          <SectionLabel>Sweeps</SectionLabel>
          <DataTable
            columns={sweepColumns}
            data={sweeps}
            getRowKey={(sweep) => sweep.id}
            onRowClick={(sweep) =>
              navigate({
                to: '/project/$projectId/experiments/sweeps/$sweepId',
                params: { projectId, sweepId: sweep.id },
              })
            }
          />
        </div>
      )}
    </div>
  )
}
