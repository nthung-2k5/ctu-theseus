import {
  Badge,
  Button,
  Checkbox,
  Group,
  Loader,
  MultiSelect,
  Paper,
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
  StatusBadge,
} from '@public/components/ui'
import { deleteRun as deleteRunRequest, getListRunsQueryKey } from '@public/lib/api/generated/training/training'
import { formatDate } from '@public/lib/format'
import { projectDetailQueryOptions, useProjectSweeps, useTrainingRuns } from '@public/lib/queries'
import { bestRun, goodness, headlineScore, scoreOf } from '@public/lib/runScore'
import type { SweepSummary, TrainingRunSummary } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useMemo, useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/training/')

const STATUSES = ['queued', 'running', 'succeeded', 'failed', 'canceled']

const SWEEP_STATUS_COLORS: Record<string, string> = { running: 'cyan', completed: 'teal', canceled: 'gray' }

export function TrainingPage() {
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

  // The filters live in the URL, like the data page's, so a filtered list survives a reload and can be shared.
  const { q = '', status, sort = 'newest', show = 'runs' } = routeApi.useSearch()
  const statuses = useMemo(() => (status ? status.split(',').filter(Boolean) : []), [status])
  const setFilter = (patch: { q?: string; status?: string[]; sort?: 'newest' | 'best'; show?: 'runs' | 'sweeps' }) =>
    navigate({
      search: (prev) => ({
        ...prev,
        ...('q' in patch ? { q: patch.q || undefined } : {}),
        ...('status' in patch ? { status: patch.status?.length ? patch.status.join(',') : undefined } : {}),
        ...('sort' in patch ? { sort: patch.sort === 'newest' ? undefined : patch.sort } : {}),
        ...('show' in patch ? { show: patch.show === 'runs' ? undefined : patch.show } : {}),
      }),
      replace: true,
    })
  const [compareIds, setCompareIds] = useState<Set<string>>(new Set())

  const toggleCompare = (runId: string) =>
    setCompareIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })

  // Every run of a project trains the same task, so they share one headline score (accuracy, mAP50, R²...).
  const headline = headlineScore(runs)
  const scoreLabel = headline?.label ?? 'Score'
  const higherIsBetter = headline?.higherIsBetter ?? true
  const bestRunId = bestRun(runs)?.id

  const visible = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return runs
      .filter((r) => (needle ? r.name.toLowerCase().includes(needle) || r.id.startsWith(needle) : true))
      .filter((r) => (statuses.length > 0 ? statuses.includes(r.status) : true))
      .sort((a, b) =>
        sort === 'best'
          ? goodness(b, higherIsBetter) - goodness(a, higherIsBetter)
          : new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
  }, [runs, q, statuses, sort, higherIsBetter])

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
    navigate({ to: '/project/$projectId/training/$runId', params: { projectId, runId } })

  const compare = () =>
    void navigate({
      to: '/project/$projectId/training/compare',
      params: { projectId },
      search: { runs: [...compareIds].join(',') },
    })

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
              <Tooltip label={`Best ${scoreLabel} in this project`}>
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
      key: 'score',
      header: scoreLabel,
      fit: true,
      render: (run) => {
        const score = scoreOf(run)
        return (
          <Text size="sm" className="tnum" c={score ? undefined : 'dimmed'}>
            {score ? score.value.toFixed(3) : '—'}
          </Text>
        )
      },
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (run) => (
        <Text size="xs" c="dimmed" className="tnum">
          {formatDate(run.createdAt)}
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
            to="/project/$projectId/training/$runId/evaluation"
            params={{ projectId, runId: run.id }}
            size="compact-xs"
            variant="light"
            disabled={run.status !== 'succeeded'}
          >
            Evaluate
          </LinkButton>
          <LinkButton
            to="/project/$projectId/training/$runId/try"
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
          {formatDate(sweep.createdAt)}
        </Text>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-3 p-3">
      <PageHeader
        title="Training"
        actions={
          <>
            <LinkButton
              to="/project/$projectId/training/new"
              params={{ projectId }}
              search={{ mode: 'sweep' }}
              variant="default"
              leftSection={<FlaskIcon size={14} />}
              disabled={!hasReadySnapshot}
            >
              New sweep
            </LinkButton>
            <LinkButton
              to="/project/$projectId/training/new"
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

      <Group gap="xs" wrap="wrap">
        {sweeps.length > 0 && (
          <SegmentedControl
            value={show}
            onChange={(v) => setFilter({ show: v as 'runs' | 'sweeps' })}
            data={[
              { value: 'runs', label: `Runs (${runs.length})` },
              { value: 'sweeps', label: `Sweeps (${sweeps.length})` },
            ]}
          />
        )}
        {show === 'runs' && runs.length > 0 && (
          <>
            <TextInput
              size="xs"
              placeholder="Search runs…"
              leftSection={<MagnifyingGlassIcon size={14} />}
              value={q}
              onChange={(e) => setFilter({ q: e.currentTarget.value })}
              style={{ flexGrow: 1, minWidth: 200, maxWidth: 320 }}
            />
            <MultiSelect
              size="xs"
              placeholder={statuses.length === 0 ? 'All statuses' : undefined}
              data={STATUSES}
              value={statuses}
              onChange={(v) => setFilter({ status: v })}
              clearable
              w={220}
            />
            <SegmentedControl
              ml="auto"
              value={sort}
              onChange={(v) => setFilter({ sort: v as 'newest' | 'best' })}
              data={[
                { value: 'newest', label: 'Newest' },
                { value: 'best', label: `Best ${scoreLabel}` },
              ]}
            />
          </>
        )}
      </Group>

      {show === 'sweeps' && sweeps.length > 0 ? (
        <DataTable
          columns={sweepColumns}
          data={sweeps}
          getRowKey={(sweep) => sweep.id}
          onRowClick={(sweep) =>
            navigate({
              to: '/project/$projectId/training/sweeps/$sweepId',
              params: { projectId, sweepId: sweep.id },
            })
          }
        />
      ) : (
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
                    to="/project/$projectId/training/new"
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
      )}

      {compareIds.size > 0 && (
        <Paper
          p="xs"
          shadow="md"
          style={{ position: 'sticky', bottom: 12, zIndex: 10 }}
          role="region"
          aria-label="Selected runs"
        >
          <Group justify="space-between" wrap="nowrap">
            <Text size="sm" className="tnum">
              {compareIds.size} run{compareIds.size === 1 ? '' : 's'} selected
              {compareIds.size < 2 && (
                <Text span c="dimmed">
                  {' '}
                  — pick one more to compare
                </Text>
              )}
            </Text>
            <Group gap="xs" wrap="nowrap">
              <Button variant="subtle" color="gray" onClick={() => setCompareIds(new Set())}>
                Clear
              </Button>
              <Button leftSection={<ScalesIcon size={14} />} disabled={compareIds.size < 2} onClick={compare}>
                Compare
              </Button>
            </Group>
          </Group>
        </Paper>
      )}
    </div>
  )
}
