/**
 * RunOverviewPanel – at-a-glance summary of a training run: when it ran, how
 * long it took, which model architecture it used, and the hyperparameters it
 * was started with. Hyperparameters come from GET /runs/:runId (the summary
 * list doesn't carry them) — fetched unconditionally since they're fixed at
 * queue time and don't change as the run progresses.
 */

import { Badge, Card, Group, SimpleGrid, Skeleton, Stack, Text, Title } from '@mantine/core'
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
  augmentations: 'Augmentation',
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

function InfoStat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
        {label}
      </Text>
      <Text size="sm" fw={500}>
        {value}
      </Text>
    </div>
  )
}

export function RunOverviewPanel({ run, task }: { run: TrainingRunSummary; task: ProjectTask }) {
  // Not gated on run status — hyperparameters are fixed at queue time.
  const { data, isLoading } = useTrainingRunDetail(run.id, true)
  const hyperparameters = (data?.run.hyperparameters ?? null) as Record<string, unknown> | null

  const descriptor = getTaskDescriptor(task)
  const encoders = descriptor.ludwig?.encoders ?? []
  const encoderId = hyperparameters?.encoderId as string | undefined
  const modelLabel = encoderId ? (encoders.find((e) => e.id === encoderId)?.label ?? encoderId) : descriptor.label

  const isActive = run.status === 'running' || run.status === 'queued'
  const startedAt = run.startedAt ? new Date(run.startedAt) : null
  const completedAt = run.completedAt ? new Date(run.completedAt) : null
  // Not started yet (still queued): no duration. Still running: duration so
  // far against wall-clock now. Otherwise: the final started→completed span.
  const durationEnd = completedAt ?? (isActive ? new Date() : null)
  const duration = startedAt && durationEnd ? formatDuration(durationEnd.getTime() - startedAt.getTime()) : null

  const configEntries = hyperparameters ? Object.entries(hyperparameters).filter(([key]) => key !== 'encoderId') : []

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="lg">
        <Group justify="space-between">
          <Title order={4}>{run.name}</Title>
          <Badge variant="light" color={STATUS_COLORS[run.status] ?? 'gray'} tt="capitalize">
            {run.status}
          </Badge>
        </Group>

        <Stack gap="sm">
          <InfoStat label="Created" value={new Date(run.createdAt).toLocaleString()} />
          <InfoStat label="Started" value={startedAt ? startedAt.toLocaleString() : '—'} />
          <InfoStat label="Completed" value={completedAt ? completedAt.toLocaleString() : '—'} />
          <InfoStat label="Duration" value={duration ?? '—'} />
        </Stack>

        <div>
          <Text size="sm" fw={600} mb="xs">
            Model
          </Text>
          {isLoading ? <Skeleton height={20} width={160} /> : <Badge variant="light">{modelLabel}</Badge>}
        </div>

        <div>
          <Text size="sm" fw={600} mb="xs">
            Configuration
          </Text>
          {isLoading ? (
            <Skeleton height={60} />
          ) : configEntries.length === 0 ? (
            <Text size="sm" c="dimmed">
              No configuration recorded for this run.
            </Text>
          ) : (
            <SimpleGrid cols={2} spacing="lg">
              {configEntries.map(([key, value]) => (
                <InfoStat key={key} label={HYPERPARAM_LABELS[key] ?? key} value={formatHyperparamValue(value)} />
              ))}
            </SimpleGrid>
          )}
        </div>
      </Stack>
    </Card>
  )
}
