import { Badge, Box, Button, Checkbox, Grid, Group, Loader, Stack, Tabs, Text, Tooltip } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  ArrowLeftIcon,
  BrainIcon,
  ChartBarIcon,
  ChartLineIcon,
  CrosshairIcon,
  FlaskIcon,
  PackageIcon,
  PlusIcon,
  ScalesIcon,
  TrashIcon,
  TrophyIcon,
} from '@phosphor-icons/react'
import { CreateRunPanel } from '@public/components/training/CreateRunPanel'
import { CreateSweepPanel, type SweepStartConfig } from '@public/components/training/CreateSweepPanel'
import { STATUS_COLORS } from '@public/components/training/constants'
import { EvaluationPanel } from '@public/components/training/EvaluationPanel'
import { RunComparisonPanel } from '@public/components/training/RunComparisonPanel'
import { RunExportPanel } from '@public/components/training/RunExportPanel'
import { RunInferencePanel } from '@public/components/training/RunInferencePanel'
import { RunOverviewPanel } from '@public/components/training/RunOverviewPanel'
import { SweepDetailPanel } from '@public/components/training/SweepDetailPanel'
import { VersionDetailPanel } from '@public/components/training/VersionDetailPanel'
import {
  confirmDelete,
  DataTable,
  type DataTableColumn,
  EmptyState,
  PageHeader,
  QueryBoundary,
} from '@public/components/ui'
import { apiErrorMessage } from '@public/lib/api/client'
import type { TrainBody } from '@public/lib/api/generated/models'
import { getCreateSweepMutationOptions } from '@public/lib/api/generated/sweeps/sweeps'
import {
  cancelRun,
  deleteRun as deleteRunRequest,
  getListRunsQueryKey,
  getStartTrainingMutationOptions,
} from '@public/lib/api/generated/training/training'
import {
  projectDetailQueryOptions,
  projectSweepsQueryOptions,
  useProjectSweeps,
  useTrainingRuns,
} from '@public/lib/queries'
import type { SweepSummary, TrainingRunSummary } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'
import { useState } from 'react'

const routeApi = getRouteApi('/_app/project/$projectId/training')

/** Mirrors the route's `tab` search-param enum (web/routes/project.route.tsx). */
const RUN_TABS = ['metrics', 'evaluation', 'export', 'inference'] as const
type RunTab = (typeof RUN_TABS)[number]

const BackButton = ({ onClick, label }: { onClick: () => void; label: string }) => (
  <Button variant="subtle" color="gray" leftSection={<ArrowLeftIcon size={16} />} onClick={onClick}>
    {label}
  </Button>
)

export function TrainingPage() {
  const { projectId } = routeApi.useParams()
  const { runId: selectedView, tab, view, sweepId: selectedSweepId } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const goToList = () => navigate({ search: {} })
  const goToCreate = () => navigate({ search: { view: 'create' } })
  const goToCreateSweep = () => navigate({ search: { view: 'sweep' } })
  const goToRun = (runId: string) => navigate({ search: { runId } })
  const goToSweep = (sweepId: string) => navigate({ search: { sweepId } })
  const setTab = (next: string | null) => {
    const tab: RunTab | undefined =
      next && next !== 'metrics' && (RUN_TABS as readonly string[]).includes(next) ? (next as RunTab) : undefined
    navigate({ search: (prev) => ({ ...prev, tab }) })
  }

  /* ── Sweeps ── */
  const { data: sweepsData } = useProjectSweeps(projectId)
  const sweeps: SweepSummary[] = sweepsData?.sweeps ?? []
  const queryClient = useQueryClient()
  const sweepsQueryKey = projectSweepsQueryOptions(projectId).queryKey

  const startSweep = useMutation({
    ...getCreateSweepMutationOptions(),
    onSuccess: ({ sweep }) => {
      queryClient.invalidateQueries({ queryKey: sweepsQueryKey })
      notifications.show({
        title: 'Sweep started',
        message: `Dispatching trials for "${sweep.name}"…`,
        color: 'blue',
      })
      goToSweep(sweep.id)
    },
    onError: (error) => {
      notifications.show({ title: 'Error', message: apiErrorMessage(error, 'Failed to start sweep'), color: 'red' })
    },
  })
  const handleStartSweep = (config: SweepStartConfig) => startSweep.mutate({ projectId, data: config })

  /* ── Fetch runs from API (polls while any run is active) ── */
  const { data: runsData, isLoading, isError, refetch } = useTrainingRuns(projectId)
  const runs: TrainingRunSummary[] = runsData?.runs ?? []

  const selectedRun = selectedView ? runs.find((r) => r.id === selectedView) : undefined

  // The run with the highest denormalized accuracy across every successful
  // evaluation in the project — not just the 3 most recent, since this list
  // holds all of them. Only meaningful once at least one run has evaluated.
  const bestRunId = runs.reduce<{ id: string; accuracy: number } | null>((best, run) => {
    const accuracy = run.evaluation?.status === 'success' ? run.evaluation.accuracy : null
    if (accuracy === null) return best
    return !best || accuracy > best.accuracy ? { id: run.id, accuracy } : best
  }, null)?.id

  /* ── Run comparison selection ── */
  const [compareIds, setCompareIds] = useState<Set<string>>(new Set())
  const [comparing, setComparing] = useState(false)
  const toggleCompare = (runId: string) =>
    setCompareIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })

  // Export and inference both need a finished model, so their tabs only open
  // for a succeeded run — and a tab that closes under us falls back to metrics
  // rather than rendering an empty panel.
  const canUseModel = selectedRun?.status === 'succeeded'
  const activeTab = tab && (tab === 'metrics' || canUseModel) ? tab : 'metrics'

  const runsQueryKey = getListRunsQueryKey(projectId)

  /* ── Start training mutation ── */
  const startTraining = useMutation({
    ...getStartTrainingMutationOptions(),
    onSuccess: ({ run }) => {
      queryClient.invalidateQueries({ queryKey: runsQueryKey })
      goToRun(run.id)
    },
  })

  /* ── Stop training mutation ── */
  const stopTraining = useMutation({
    mutationFn: async (runId: string) => {
      return cancelRun(runId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: runsQueryKey })
    },
  })

  /* ── Delete run mutation ── */
  const deleteRun = useMutation({
    mutationFn: async (runId: string) => {
      return deleteRunRequest(runId)
    },
    onSuccess: (_data, runId) => {
      queryClient.invalidateQueries({ queryKey: runsQueryKey })
      if (selectedView === runId) goToList()
    },
  })

  /* ── Handlers ── */
  const handleStartTraining = (config: { name: string; datasetVersionId: string; hyperparameters: unknown }) => {
    startTraining.mutate({ projectId, data: config as TrainBody })
  }

  const handleStopTraining = () => {
    if (!selectedRun) return
    stopTraining.mutate(selectedRun.id)
  }

  const handleDeleteRun = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>, run: TrainingRunSummary) => {
    e.stopPropagation()
    const isActive = run.status === 'running' || run.status === 'queued'
    confirmDelete({
      title: 'Delete training run',
      message: (
        <>
          Are you sure you want to delete <strong>{run.name}</strong>?{' '}
          {isActive
            ? 'This will stop the in-progress training and permanently remove its metrics and exports.'
            : 'This permanently removes its metrics and exports.'}{' '}
          This cannot be undone.
        </>
      ),
      onConfirm: () => deleteRun.mutate(run.id),
    })
  }

  const sweepColumns: DataTableColumn<SweepSummary>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (sweep) => (
        <Text size="sm" fw={600}>
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
        <Badge
          variant="light"
          color={sweep.status === 'running' ? 'blue' : sweep.status === 'completed' ? 'teal' : 'gray'}
          tt="capitalize"
        >
          {sweep.status === 'running' && <Loader size={8} color="blue" mr={4} />}
          {sweep.status}
        </Badge>
      ),
    },
    {
      key: 'trials',
      header: 'Trials',
      fit: true,
      render: (sweep) => (
        <Text size="sm" c="dimmed">
          {sweep.completedTrialCount} / {sweep.trialCount}
        </Text>
      ),
    },
    {
      key: 'createdAt',
      header: 'Created',
      fit: true,
      render: (sweep) => (
        <Text size="sm" c="dimmed">
          {new Date(sweep.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
  ]

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
        />
      ),
    },
    {
      key: 'name',
      header: 'Name',
      render: (run) => (
        <Group gap={6} wrap="nowrap">
          <Text size="sm" fw={600}>
            {run.name}
          </Text>
          {run.id === bestRunId && (
            <Tooltip label="Highest accuracy in this project">
              <TrophyIcon size={14} weight="fill" color="var(--mantine-color-yellow-6)" />
            </Tooltip>
          )}
        </Group>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      fit: true,
      render: (run) => {
        const isActive = run.status === 'running' || run.status === 'queued'
        return (
          <Badge variant="light" color={STATUS_COLORS[run.status] ?? 'gray'} tt="capitalize">
            {isActive && <Loader size={8} color={STATUS_COLORS[run.status] ?? 'gray'} mr={4} />}
            {run.status}
          </Badge>
        )
      },
    },
    {
      key: 'accuracy',
      header: 'Accuracy',
      fit: true,
      render: (run) => (
        <Text size="sm" c={run.evaluation?.accuracy != null ? undefined : 'dimmed'}>
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
        <Text size="sm" c="dimmed">
          {new Date(run.createdAt).toLocaleDateString()}
        </Text>
      ),
    },
    {
      key: 'actions',
      header: '',
      fit: true,
      render: (run) => (
        <Button
          size="xs"
          variant="subtle"
          color="red"
          leftSection={<TrashIcon size={14} />}
          onClick={(e) => handleDeleteRun(e, run)}
        >
          Delete
        </Button>
      ),
    },
  ]

  /* ── Run detail page ── */
  if (selectedView) {
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader
            title={selectedRun?.name ?? 'Training run'}
            description="Monitor progress, then export or test the model this run produced"
            actions={<BackButton onClick={goToList} label="Back to runs" />}
          />

          {selectedRun ? (
            <Grid>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <RunOverviewPanel run={selectedRun} task={project.task} />
              </Grid.Col>
              <Grid.Col span={{ base: 12, md: 8 }}>
                <Tabs value={activeTab} onChange={setTab}>
                  <Tabs.List mb="md">
                    <Tabs.Tab value="metrics" leftSection={<ChartLineIcon size={14} />}>
                      Metrics
                    </Tabs.Tab>
                    <Tabs.Tab
                      value="evaluation"
                      leftSection={<ChartBarIcon size={14} />}
                      disabled={!canUseModel}
                      title={canUseModel ? undefined : 'Available once the run succeeds'}
                    >
                      Evaluation
                    </Tabs.Tab>
                    <Tabs.Tab
                      value="inference"
                      leftSection={<CrosshairIcon size={14} />}
                      disabled={!canUseModel}
                      title={canUseModel ? undefined : 'Available once the run succeeds'}
                    >
                      Inference
                    </Tabs.Tab>
                    <Tabs.Tab
                      value="export"
                      leftSection={<PackageIcon size={14} />}
                      disabled={!canUseModel}
                      title={canUseModel ? undefined : 'Available once the run succeeds'}
                    >
                      Export
                    </Tabs.Tab>
                  </Tabs.List>

                  <Tabs.Panel value="metrics">
                    <VersionDetailPanel
                      projectId={projectId}
                      run={selectedRun}
                      onStop={
                        selectedRun.status === 'running' || selectedRun.status === 'queued'
                          ? handleStopTraining
                          : undefined
                      }
                    />
                  </Tabs.Panel>
                  <Tabs.Panel value="evaluation">
                    <EvaluationPanel run={selectedRun} projectId={projectId} modality={project.dataset?.modality} />
                  </Tabs.Panel>
                  <Tabs.Panel value="export">
                    <RunExportPanel run={selectedRun} />
                  </Tabs.Panel>
                  <Tabs.Panel value="inference">
                    <RunInferencePanel projectId={projectId} run={selectedRun} />
                  </Tabs.Panel>
                </Tabs>
              </Grid.Col>
            </Grid>
          ) : (
            <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
              <EmptyState icon={BrainIcon} title="Run not found" description="It may have been deleted." />
            </QueryBoundary>
          )}
        </Stack>
      </Box>
    )
  }

  /* ── New run page ── */
  if (view === 'create') {
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader
            title="New Training Run"
            description="Configure and start a new training run"
            actions={<BackButton onClick={goToList} label="Back to runs" />}
          />
          <CreateRunPanel project={project} onStartTraining={handleStartTraining} />
        </Stack>
      </Box>
    )
  }

  /* ── New sweep page ── */
  if (view === 'sweep') {
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader
            title="New Sweep"
            description="Search over a handful of hyperparameters by dispatching one training run per combination"
            actions={<BackButton onClick={goToList} label="Back to runs" />}
          />
          <CreateSweepPanel project={project} onStartSweep={handleStartSweep} />
        </Stack>
      </Box>
    )
  }

  /* ── Sweep detail page ── */
  if (selectedSweepId) {
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader title="Sweep" description="One training run per hyperparameter combination" />
          <SweepDetailPanel sweepId={selectedSweepId} onBack={goToList} onOpenRun={goToRun} />
        </Stack>
      </Box>
    )
  }

  /* ── Run comparison page ── */
  if (comparing) {
    const comparedRuns = runs.filter((r) => compareIds.has(r.id))
    return (
      <Box>
        <Stack gap="xl">
          <PageHeader title="Compare Runs" description="Side-by-side accuracy, macro F1, and hyperparameters." />
          <RunComparisonPanel
            runs={comparedRuns}
            onBack={() => {
              setComparing(false)
              setCompareIds(new Set())
            }}
          />
        </Stack>
      </Box>
    )
  }

  /* ── Run list page ── */
  return (
    <Box>
      <Stack gap="xl">
        <PageHeader
          title="Training"
          description="Manage training runs, monitor progress, then export or test the model a run produced"
          actions={
            <Group gap="sm">
              {compareIds.size > 0 && (
                <Button
                  variant="light"
                  leftSection={<ScalesIcon size={16} />}
                  disabled={compareIds.size < 2}
                  onClick={() => setComparing(true)}
                >
                  Compare ({compareIds.size})
                </Button>
              )}
              <Button variant="light" leftSection={<FlaskIcon size={16} />} onClick={goToCreateSweep}>
                New Sweep
              </Button>
              <Button leftSection={<PlusIcon size={16} />} onClick={goToCreate}>
                New Run
              </Button>
            </Group>
          }
        />

        {sweeps.length > 0 && (
          <Stack gap="xs">
            <Text size="sm" fw={600}>
              Sweeps
            </Text>
            <DataTable
              columns={sweepColumns}
              data={sweeps}
              getRowKey={(sweep) => sweep.id}
              onRowClick={(sweep) => goToSweep(sweep.id)}
            />
          </Stack>
        )}

        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {runs.length === 0 ? (
            <EmptyState icon={BrainIcon} title="No runs yet" description="Create your first training run" />
          ) : (
            <DataTable
              columns={runColumns}
              data={runs}
              getRowKey={(run) => run.id}
              onRowClick={(run) => goToRun(run.id)}
            />
          )}
        </QueryBoundary>
      </Stack>
    </Box>
  )
}
