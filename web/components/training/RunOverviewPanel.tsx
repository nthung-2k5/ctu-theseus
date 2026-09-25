/**
 * RunOverviewPanel – at-a-glance summary of a training run: when it ran, how
 * long it took, which model architecture it used, and the hyperparameters it
 * was started with. Hyperparameters come from GET /runs/:runId (the summary
 * list doesn't carry them) — fetched unconditionally since they're fixed at
 * queue time and don't change as the run progresses.
 */

import { Badge, Group, Paper, SimpleGrid, Skeleton, Stack, Table, Text } from '@mantine/core'
import { SectionLabel, StatusBadge } from '@public/components/ui'
import { useListProjectTrainingBackends } from '@public/lib/api/generated/training/training'
import { useTrainingRunDetail } from '@public/lib/queries'
import { getTaskDescriptor } from '@public/lib/tasks'
import type { ProjectTask, TrainingRunSummary } from '@public/store/types'
import type { ReactNode } from 'react'
import { STATUS_COLORS } from './constants'

const HYPERPARAM_LABELS: Record<string, string> = {
  epochs: 'Epochs',
  batchSize: 'Batch Size',
  learningRate: 'Learning Rate',
  earlyStopPatience: 'Early Stop Patience',
  useClassWeights: 'Class Weighting',
  // No longer a training option (augmentation is set when a snapshot is created), but runs started
  // before that change still carry it in their hyperparameters.
  augmentations: 'Augmentation (legacy)',
}

function formatHyperparamValue(value: unknown): string {
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : 'None'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  return String(value)
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(totalSeconds / 3600)
  const m = Math.floor((totalSeconds % 3600) / 60)
  const s = totalSeconds % 60
  const parts: string[] = []
  if (h > 0) parts.push(`${h}h`)
  if (h > 0 || m > 0) parts.push(`${m}m`)
  parts.push(`${s}s`)
  return parts.join(' ')
}

function KeyValueTable({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <Table verticalSpacing={4} withRowBorders={false} className="tnum">
      <Table.Tbody>
        {rows.map(([label, value]) => (
          <Table.Tr key={label}>
            <Table.Td w="45%">
              <Text size="xs" c="dimmed">
                {label}
              </Text>
            </Table.Td>
            <Table.Td>
              <Text size="sm">{value}</Text>
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  )
}

export function RunOverviewPanel({
  run,
  task,
  projectId,
}: {
  run: TrainingRunSummary
  task: ProjectTask
  projectId: string
}) {
  // Not gated on run status — hyperparameters are fixed at queue time.
  const { data, isLoading } = useTrainingRunDetail(run.id, true)
  const hyperparameters = (data?.run.hyperparameters ?? null) as Record<string, unknown> | null
  // Model choices are per trainer backend now (GET /training-backends), not a static per-task
  // list — a run's own model id might belong to any of the backends this project could train with.
  const { data: backendsData } = useListProjectTrainingBackends(projectId)
  const models = backendsData?.backends.flatMap((b) => b.models) ?? []

  const descriptor = getTaskDescriptor(task)
  const encoderId = hyperparameters?.encoderId as string | undefined
  const modelLabel = encoderId ? (models.find((m) => m.id === encoderId)?.label ?? encoderId) : descriptor.label

  const isActive = run.status === 'running' || run.status === 'queued'
  const startedAt = run.startedAt ? new Date(run.startedAt) : null
  const completedAt = run.completedAt ? new Date(run.completedAt) : null
  // Not started yet (still queued): no duration. Still running: duration so
  // far against wall-clock now. Otherwise: the final started→completed span.
  const durationEnd = completedAt ?? (isActive ? new Date() : null)
  const duration = startedAt && durationEnd ? formatDuration(durationEnd.getTime() - startedAt.getTime()) : null

  const configEntries = hyperparameters ? Object.entries(hyperparameters).filter(([key]) => key !== 'encoderId') : []

  return (
    <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
      <Paper p="md">
        <Stack gap="xs">
          <Group justify="space-between">
            <SectionLabel>Run</SectionLabel>
            <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
          </Group>
          <KeyValueTable
            rows={[
              ['Created', new Date(run.createdAt).toLocaleString()],
              ['Started', startedAt ? startedAt.toLocaleString() : '—'],
              ['Completed', completedAt ? completedAt.toLocaleString() : '—'],
              ['Duration', duration ?? '—'],
              [
                'Model',
                isLoading ? <Skeleton key="m" height={16} width={140} /> : <Badge key="model">{modelLabel}</Badge>,
              ],
            ]}
          />
        </Stack>
      </Paper>

      <Paper p="md">
        <Stack gap="xs">
          <SectionLabel>Hyperparameters</SectionLabel>
          {isLoading ? (
            <Skeleton height={60} />
          ) : configEntries.length === 0 ? (
            <Text size="sm" c="dimmed">
              No configuration recorded for this run.
            </Text>
          ) : (
            <KeyValueTable
              rows={configEntries.map(([key, value]) => [HYPERPARAM_LABELS[key] ?? key, formatHyperparamValue(value)])}
            />
          )}
        </Stack>
      </Paper>
    </SimpleGrid>
  )
}
