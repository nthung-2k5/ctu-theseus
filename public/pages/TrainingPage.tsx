import { Badge, Box, Button, Card, Grid, Group, ScrollArea, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { BrainIcon, PlusIcon } from '@phosphor-icons/react'
import { CreateVersionPanel } from '@public/components/training/CreateVersionPanel'
import {
  createMockRuns,
  getVariantLabel,
  STATUS_COLORS,
  type TrainingRunData,
  type TrainingVersionConfig,
} from '@public/components/training/constants'
import { VersionDetailPanel } from '@public/components/training/VersionDetailPanel'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useParams } from 'wouter'

export function TrainingPage() {
  const params = useParams<{ id: string }>()
  const projectId = params.id

  const [runs, setRuns] = useState<TrainingRunData[]>(() => createMockRuns())
  const [selectedView, setSelectedView] = useState<'create' | string>('create')
  const simulationsRef = useRef(new Map<string, ReturnType<typeof setInterval>>())

  const selectedRun = selectedView !== 'create' ? runs.find((r) => r.id === selectedView) : undefined

  /* ── Simulate training progress ── */
  const simulateTraining = useCallback((runId: string) => {
    let currentEpoch = 0
    const k = 0.055

    const interval = setInterval(() => {
      setRuns((prev) => {
        const run = prev.find((r) => r.id === runId)
        if (!run || run.status !== 'training') {
          clearInterval(interval)
          simulationsRef.current.delete(runId)
          return prev
        }

        currentEpoch++
        const t = currentEpoch
        const newMetric = {
          epoch: t,
          trainLoss: +(2.5 * Math.exp(-k * t) + 0.14 + (Math.random() - 0.5) * 0.04).toFixed(4),
          valLoss: +(2.8 * Math.exp(-(k - 0.01) * t) + 0.24 + (Math.random() - 0.5) * 0.06).toFixed(4),
          accuracy: +Math.min(0.94, 0.28 + 0.66 * (1 - Math.exp(-0.08 * t)) + (Math.random() - 0.5) * 0.015).toFixed(4),
          mAP: +Math.min(0.91, 0.22 + 0.62 * (1 - Math.exp(-0.07 * t)) + (Math.random() - 0.5) * 0.02).toFixed(4),
        }

        const isComplete = currentEpoch >= run.epochs

        if (isComplete) {
          clearInterval(interval)
          simulationsRef.current.delete(runId)
        }

        return prev.map((r) =>
          r.id !== runId
            ? r
            : {
                ...r,
                status: isComplete ? ('completed' as const) : ('training' as const),
                metrics: [...r.metrics, newMetric],
                ...(isComplete ? { completedAt: new Date() } : {}),
              },
        )
      })
    }, 800)

    simulationsRef.current.set(runId, interval)
  }, [])

  /* ── Cleanup simulations on unmount ── */
  useEffect(() => {
    return () => {
      for (const interval of simulationsRef.current.values()) clearInterval(interval)
    }
  }, [])

  /* ── Start training handler ── */
  const handleStartTraining = (config: TrainingVersionConfig) => {
    const label = getVariantLabel(config.modelId, config.variantId)
    const existingCount = runs.filter((r) => r.variantId === config.variantId).length
    const name = `${label} v${existingCount + 1}`

    const newRun: TrainingRunData = {
      ...config,
      id: `run-${Date.now()}`,
      name,
      status: 'queued',
      metrics: [],
      createdAt: new Date(),
    }

    setRuns((prev) => [newRun, ...prev])
    setSelectedView(newRun.id)

    // Simulate: queued → training after 1.5s
    setTimeout(() => {
      setRuns((prev) => prev.map((r) => (r.id === newRun.id ? { ...r, status: 'training', startedAt: new Date() } : r)))
      simulateTraining(newRun.id)
    }, 1500)
  }

  /* ── Stop training handler ── */
  const handleStopTraining = () => {
    if (!selectedRun) return
    const interval = simulationsRef.current.get(selectedRun.id)
    if (interval) {
      clearInterval(interval)
      simulationsRef.current.delete(selectedRun.id)
    }
    setRuns((prev) =>
      prev.map((r) =>
        r.id === selectedRun.id
          ? { ...r, status: 'failed' as const, error: 'Training stopped by user', completedAt: new Date() }
          : r,
      ),
    )
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
                  {runs.map((run) => (
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
                            {run.createdAt.toLocaleDateString()}
                          </Text>
                        </div>
                        <Badge size="xs" variant="light" color={STATUS_COLORS[run.status]} tt="capitalize">
                          {run.status}
                        </Badge>
                      </Group>
                      {run.status === 'completed' && run.metrics.length > 0 && (
                        <Text size="xs" c="dimmed" mt={4}>
                          Accuracy: {(run.metrics[run.metrics.length - 1].accuracy * 100).toFixed(1)}%
                        </Text>
                      )}
                    </Card>
                  ))}

                  {runs.length === 0 && (
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
                onStop={selectedRun.status === 'training' ? handleStopTraining : undefined}
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
