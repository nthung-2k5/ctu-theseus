import { Badge, Box, Button, Card, Grid, Group, Loader, ScrollArea, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { BrainIcon, PlusIcon } from '@phosphor-icons/react'
import { CreateVersionPanel } from '@public/components/training/CreateVersionPanel'
import { STATUS_COLORS } from '@public/components/training/constants'
import { VersionDetailPanel } from '@public/components/training/VersionDetailPanel'
import { api } from '@public/lib/api'
import { useEdenMutation } from '@public/lib/eden-query'
import { training, useTrainingRuns } from '@public/queries/training'
import type { TrainingRunSummary, TrainingVersionConfig } from '@public/store/types'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useParams } from 'wouter'

export function TrainingPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id

  const queryClient = useQueryClient()
  const [selectedView, setSelectedView] = useState<'create' | string>('create')

  /* ── Fetch runs from API (polls while any run is active) ── */
  const { data: runsData, isLoading } = useTrainingRuns(projectId)
  const runs: TrainingRunSummary[] = runsData?.runs ?? []

  const selectedRun = selectedView !== 'create' ? runs.find((r) => r.id === selectedView) : undefined

  /* ── Start training mutation ── */
  const startTraining = useEdenMutation(
    (config: TrainingVersionConfig) => api.projects({ projectId }).train.post(config),
    [training.runs(projectId).queryKey],
  )

  /* ── Stop training mutation ── */
  const stopTraining = useEdenMutation(
    (runId: string) => api.runs({ runId }).stop.post(),
    [training.runs(projectId).queryKey],
  )

  /* ── Handlers ── */
  const handleStartTraining = (config: TrainingVersionConfig) => {
    startTraining.mutate(config, {
      onSuccess: () => {
        // Refetch runs to get the new run
        queryClient.invalidateQueries({ queryKey: training.runs(projectId).queryKey })
      },
    })
  }

  const handleStopTraining = () => {
    if (!selectedRun) return
    stopTraining.mutate(selectedRun.id, {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: training.runs(projectId).queryKey })
      },
    })
  }

  return (
    <Box>
      <Stack gap="xl">
        {/* Page header */}
        <div>
          <Title order={2}>Training</Title>
          <Text size="sm" c="dimmed" mt={4}>
            Manage training versions and monitor progress
          </Text>
        </div>

        <Grid gap="lg">
          {/* ── Left sidebar: version list ── */}
          <Grid.Col span={{ base: 12, md: 3 }}>
            <Stack gap="sm">
              <Button
                me="sm"
                leftSection={<PlusIcon size={16} />}
                variant={selectedView === 'create' ? 'filled' : 'light'}
                onClick={() => setSelectedView('create')}
              >
                New Version
              </Button>

              <ScrollArea h="calc(100vh - 260px)" offsetScrollbars>
                <Stack gap="xs">
                  {isLoading && (
                    <Card withBorder p="md" radius="md" ta="center">
                      <Loader size="sm" />
                    </Card>
                  )}

                  {runs.map((run, index) => {
                    const versionNumber = runs.length - index
                    const isActive = run.status === 'training' || run.status === 'queued' || run.status === 'preparing'

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
                              Version {versionNumber}
                            </Text>
                            <Text size="xs" c="dimmed">
                              {run.modelName}
                            </Text>
                            <Text size="xs" c="dimmed">
                              {new Date(run.createdAt).toLocaleDateString()}
                            </Text>
                          </div>
                          <Badge size="xs" variant="light" color={STATUS_COLORS[run.status]} tt="capitalize">
                            {isActive && <Loader size={8} color={STATUS_COLORS[run.status]} mr={4} />}
                            {run.status}
                          </Badge>
                        </Group>
                      </Card>
                    )
                  })}

                  {!isLoading && runs.length === 0 && (
                    <Card withBorder p="md" radius="md" ta="center">
                      <Stack align="center" gap="xs">
                        <ThemeIcon size="xl" variant="light" color="gray" radius="xl">
                          <BrainIcon size={24} />
                        </ThemeIcon>
                        <Text size="sm" c="dimmed">
                          No versions yet
                        </Text>
                        <Text size="xs" c="dimmed">
                          Create your first training version
                        </Text>
                      </Stack>
                    </Card>
                  )}
                </Stack>
              </ScrollArea>
            </Stack>
          </Grid.Col>

          {/* ── Right panel ── */}
          <Grid.Col span={{ base: 12, md: 9 }}>
            {selectedView === 'create' ? (
              <CreateVersionPanel projectId={projectId} onStartTraining={handleStartTraining} />
            ) : selectedRun ? (
              <VersionDetailPanel
                run={selectedRun}
                versionNumber={runs.length - runs.indexOf(selectedRun)}
                onStop={
                  selectedRun.status === 'training' || selectedRun.status === 'queued' ? handleStopTraining : undefined
                }
              />
            ) : (
              <Card withBorder p="xl" radius="md" ta="center">
                <Stack align="center" gap="sm">
                  <ThemeIcon size={48} variant="light" color="primary" radius="xl">
                    <BrainIcon size={28} />
                  </ThemeIcon>
                  <Title order={5}>Select a version</Title>
                  <Text size="sm" c="dimmed">
                    Choose a training version from the sidebar or create a new one
                  </Text>
                </Stack>
              </Card>
            )}
          </Grid.Col>
        </Grid>
      </Stack>
    </Box>
  )
}
