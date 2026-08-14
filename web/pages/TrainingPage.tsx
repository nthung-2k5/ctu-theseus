import { Badge, Box, Button, Card, Grid, Group, Loader, ScrollArea, Stack, Text } from '@mantine/core'
import { BrainIcon, PlusIcon } from '@phosphor-icons/react'
import { CreateRunPanel } from '@public/components/training/CreateRunPanel'
import { STATUS_COLORS } from '@public/components/training/constants'
import { VersionDetailPanel } from '@public/components/training/VersionDetailPanel'
import { EmptyState, PageHeader } from '@public/components/ui'
import { rest, useEden } from '@public/lib/api'
import { projectDetailQueryOptions, useTrainingRuns } from '@public/lib/queries'
import type { TrainingRunSummary } from '@public/store/types'
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query'
import { getRouteApi } from '@tanstack/react-router'

const routeApi = getRouteApi('/_app/project/$projectId/training')

export function TrainingPage() {
  const { projectId } = routeApi.useParams()
  const { runId: selectedView } = routeApi.useSearch()
  const navigate = routeApi.useNavigate()
  const {
    data: { project },
  } = useSuspenseQuery(projectDetailQueryOptions(projectId))
  const setSelectedView = (runId: string | undefined) => navigate({ search: (prev) => ({ ...prev, runId }) })

  /* ── Fetch runs from API (polls while any run is active) ── */
  const { data: runsData, isLoading } = useTrainingRuns(projectId)
  const runs: TrainingRunSummary[] = runsData?.runs ?? []

  const selectedRun = selectedView ? runs.find((r) => r.id === selectedView) : undefined

  const eden = useEden()
  const queryClient = useQueryClient()
  const runsQueryKey = eden.api.projects({ projectId }).runs.get.queryKey()

  /* ── Start training mutation ── */
  const startTraining = useMutation({
    ...eden.api.projects({ projectId }).train.post.mutationOptions(),
    onSuccess: ({ run }) => {
      queryClient.invalidateQueries({ queryKey: runsQueryKey })
      setSelectedView(run.id)
    },
  })

  /* ── Stop training mutation ── */
  const stopTraining = useMutation({
    mutationFn: async (runId: string) => {
      const { data, error } = await rest.runs({ runId }).cancel.post()
      if (error) throw error
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: runsQueryKey })
    },
  })

  /* ── Handlers ── */
  const handleStartTraining = (config: { name: string; datasetVersionId: string; hyperparameters: unknown }) => {
    startTraining.mutate(config)
  }

  const handleStopTraining = () => {
    if (!selectedRun) return
    stopTraining.mutate(selectedRun.id)
  }

  return (
    <Box>
      <Stack gap="xl">
        <PageHeader title="Training" description="Manage training runs and monitor progress" />

        <Grid gap="lg">
          {/* ── Left sidebar: run list ── */}
          <Grid.Col span={{ base: 12, md: 3 }}>
            <Stack gap="sm">
              <Button
                me="sm"
                leftSection={<PlusIcon size={16} />}
                variant={!selectedView ? 'filled' : 'light'}
                onClick={() => setSelectedView(undefined)}
              >
                New Run
              </Button>

              <ScrollArea h="calc(100vh - 260px)" offsetScrollbars>
                <Stack gap="xs">
                  {isLoading && (
                    <Card withBorder p="md" radius="md" ta="center">
                      <Loader size="sm" />
                    </Card>
                  )}

                  {runs.map((run) => {
                    const isActive = run.status === 'running' || run.status === 'queued'

                    return (
                      <Card
                        key={run.id}
                        withBorder
                        p="sm"
                        radius="md"
                        style={{
                          cursor: 'pointer',
                          outline:
                            selectedView === run.id
                              ? '2px solid var(--mantine-primary-color-5)'
                              : '2px solid transparent',
                          outlineOffset: -2,
                        }}
                        onClick={() => setSelectedView(run.id)}
                      >
                        <Group justify="space-between" wrap="nowrap">
                          <div style={{ minWidth: 0 }}>
                            <Text size="sm" fw={600} truncate="end">
                              {run.name}
                            </Text>
                            <Text size="xs" c="dimmed">
                              {new Date(run.createdAt).toLocaleDateString()}
                            </Text>
                          </div>
                          <Badge size="xs" variant="light" color={STATUS_COLORS[run.status] ?? 'gray'} tt="capitalize">
                            {isActive && <Loader size={8} color={STATUS_COLORS[run.status] ?? 'gray'} mr={4} />}
                            {run.status}
                          </Badge>
                        </Group>
                      </Card>
                    )
                  })}

                  {!isLoading && runs.length === 0 && (
                    <EmptyState
                      icon={BrainIcon}
                      title="No runs yet"
                      description="Create your first training run"
                      compact
                    />
                  )}
                </Stack>
              </ScrollArea>
            </Stack>
          </Grid.Col>

          {/* ── Right panel ── */}
          <Grid.Col span={{ base: 12, md: 9 }}>
            {!selectedView ? (
              <CreateRunPanel project={project} onStartTraining={handleStartTraining} />
            ) : selectedRun ? (
              <VersionDetailPanel
                projectId={projectId}
                run={selectedRun}
                onStop={
                  selectedRun.status === 'running' || selectedRun.status === 'queued' ? handleStopTraining : undefined
                }
              />
            ) : (
              <EmptyState
                icon={BrainIcon}
                title="Select a run"
                description="Choose a training run from the sidebar or create a new one"
              />
            )}
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}
