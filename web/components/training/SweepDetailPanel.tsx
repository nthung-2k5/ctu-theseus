/**
 * SweepDetailPanel – trial leaderboard for one sweep, plus a
 * hyperparameter-vs-accuracy scatter for whichever swept knob is numeric.
 * Every trial is an ordinary training run (see server/lib/sweep.ts), so
 * this is the Phase 1 evaluation data (accuracy/macroF1, already
 * denormalized onto the run) filtered down to one sweep's runs — no
 * separate metrics pipeline for sweeps.
 */

import { ScatterChart } from '@mantine/charts'
import { Badge, Button, Card, Group, Select, Stack, Table, Text, Title } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { ArrowLeftIcon, StopIcon, TrophyIcon } from '@phosphor-icons/react'
import { EmptyState, QueryBoundary, StatusBadge } from '@public/components/ui'
import { cancelSweep as cancelSweepRequest } from '@public/lib/api/generated/sweeps/sweeps'
import { sweepDetailQueryOptions, useSweepDetail } from '@public/lib/queries'
import type { SweepSearchSpace, SweepTrial } from '@public/store/types'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { STATUS_COLORS } from './constants'

const SWEEP_STATUS_COLORS: Record<string, string> = {
  running: 'blue',
  completed: 'teal',
  canceled: 'gray',
}

function formatHyperparamValue(value: unknown): string {
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : 'None'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  return String(value)
}

function trialHeadlineMetric(trial: SweepTrial): number | null {
  if (trial.evaluation?.status !== 'success') return null
  return trial.evaluation.accuracy ?? trial.evaluation.macroF1 ?? null
}

export function SweepDetailPanel({
  sweepId,
  onBack,
  onOpenRun,
}: {
  sweepId: string
  onBack: () => void
  onOpenRun: (runId: string) => void
}) {
  const { data, isLoading, isError, refetch } = useSweepDetail(sweepId)
  const queryClient = useQueryClient()
  const [scatterKnob, setScatterKnob] = useState<string | null>(null)

  const cancelSweep = useMutation({
    mutationFn: async () => {
      await cancelSweepRequest(sweepId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sweepDetailQueryOptions(sweepId).queryKey })
      notifications.show({ title: 'Sweep canceled', message: 'Every unfinished trial was stopped.', color: 'gray' })
    },
    onError: () => notifications.show({ title: 'Error', message: 'Failed to cancel sweep', color: 'red' }),
  })

  const sweep = data?.sweep
  const trials: SweepTrial[] = data?.trials ?? []
  const searchSpace = (sweep?.searchSpace ?? {}) as SweepSearchSpace

  // Only a numeric knob can drive a scatter x-axis — encoderId is categorical.
  const numericKnobs = Object.keys(searchSpace).filter((key) => key !== 'encoderId')
  const effectiveKnob = scatterKnob && numericKnobs.includes(scatterKnob) ? scatterKnob : (numericKnobs[0] ?? null)

  const bestTrialId = useMemo(() => {
    let best: { id: string; metric: number } | null = null
    for (const trial of trials) {
      const metric = trialHeadlineMetric(trial)
      if (metric === null) continue
      if (!best || metric > best.metric) best = { id: trial.id, metric }
    }
    return best?.id
  }, [trials])

  const scatterData = useMemo(() => {
    if (!effectiveKnob) return []
    const knob = effectiveKnob
    const points: Record<string, number>[] = []
    for (const trial of trials) {
      const hyperparameters = trial.hyperparameters as Record<string, unknown> | null
      const x = hyperparameters?.[knob]
      const y = trialHeadlineMetric(trial)
      if (typeof x !== 'number' || y === null) continue
      points.push({ [knob]: x, accuracy: y })
    }
    return [{ color: 'blue.6', name: 'Trials', data: points }]
  }, [trials, effectiveKnob])

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="lg">
        <Group justify="space-between">
          <Group gap="sm">
            <Button variant="subtle" color="gray" size="xs" leftSection={<ArrowLeftIcon size={14} />} onClick={onBack}>
              Back to sweeps
            </Button>
            <Title order={5}>{sweep?.name ?? 'Sweep'}</Title>
            {sweep && (
              <Badge variant="light" color={SWEEP_STATUS_COLORS[sweep.status] ?? 'gray'} tt="capitalize">
                {sweep.status}
              </Badge>
            )}
          </Group>
          {sweep?.status === 'running' && (
            <Button
              size="xs"
              color="red"
              variant="light"
              leftSection={<StopIcon weight="fill" size={14} />}
              loading={cancelSweep.isPending}
              onClick={() => cancelSweep.mutate()}
            >
              Cancel Sweep
            </Button>
          )}
        </Group>

        <QueryBoundary isLoading={isLoading} isError={isError} onRetry={() => refetch()}>
          {trials.length === 0 ? (
            <EmptyState title="No trials yet" description="This sweep hasn't dispatched any trials." compact />
          ) : (
            <Stack gap="lg">
              {numericKnobs.length > 0 && scatterData[0]?.data.length > 0 && (
                <div>
                  <Group justify="space-between" mb="xs">
                    <Text size="sm" fw={600}>
                      Accuracy vs. {effectiveKnob}
                    </Text>
                    {numericKnobs.length > 1 && (
                      <Select
                        size="xs"
                        w={180}
                        data={numericKnobs}
                        value={effectiveKnob}
                        onChange={setScatterKnob}
                        allowDeselect={false}
                      />
                    )}
                  </Group>
                  {effectiveKnob && (
                    <ScatterChart
                      h={220}
                      data={scatterData}
                      dataKey={{ x: effectiveKnob, y: 'accuracy' }}
                      withLegend={false}
                    />
                  )}
                </div>
              )}

              <Table.ScrollContainer minWidth={640}>
                <Table verticalSpacing="sm" highlightOnHover>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Trial</Table.Th>
                      <Table.Th>Status</Table.Th>
                      <Table.Th>Accuracy</Table.Th>
                      <Table.Th>Macro F1</Table.Th>
                      <Table.Th>Hyperparameters</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {trials.map((trial) => {
                      const hyperparameters = trial.hyperparameters as Record<string, unknown> | null
                      return (
                        <Table.Tr key={trial.id} onClick={() => onOpenRun(trial.id)} style={{ cursor: 'pointer' }}>
                          <Table.Td>
                            <Group gap={6} wrap="nowrap">
                              <Text size="sm" fw={600}>
                                {trial.name}
                              </Text>
                              {trial.id === bestTrialId && (
                                <TrophyIcon size={14} weight="fill" color="var(--mantine-color-yellow-6)" />
                              )}
                            </Group>
                          </Table.Td>
                          <Table.Td>
                            <StatusBadge value={trial.status} colorMap={STATUS_COLORS} size="sm" />
                          </Table.Td>
                          <Table.Td>
                            {trial.evaluation?.accuracy != null ? trial.evaluation.accuracy.toFixed(3) : '—'}
                          </Table.Td>
                          <Table.Td>
                            {trial.evaluation?.macroF1 != null ? trial.evaluation.macroF1.toFixed(3) : '—'}
                          </Table.Td>
                          <Table.Td>
                            <Text size="xs" c="dimmed" maw={280}>
                              {hyperparameters
                                ? Object.entries(hyperparameters)
                                    .map(([key, value]) => `${key}: ${formatHyperparamValue(value)}`)
                                    .join(' · ')
                                : '—'}
                            </Text>
                          </Table.Td>
                        </Table.Tr>
                      )
                    })}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            </Stack>
          )}
        </QueryBoundary>
      </Stack>
    </Card>
  )
}
