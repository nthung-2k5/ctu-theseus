/**
 * RunComparisonPanel – side-by-side comparison of a handful of runs: status,
 * accuracy/macro-F1 (denormalized onto TrainingRunSummary from the run's
 * evaluation report — see GET /projects/:projectId/runs in
 * server/routes/training.ts), and the hyperparameters each was started with.
 * Hyperparameters aren't in the run list response (fixed at queue time, only
 * worth fetching for the handful of runs actually being compared), so each
 * row fetches its own run detail — mirrors RunOverviewPanel's same tradeoff.
 */

import { Badge, Button, Card, Group, Table, Text, Title } from '@mantine/core'
import { ArrowLeftIcon } from '@phosphor-icons/react'
import { useTrainingRunDetail } from '@public/lib/queries'
import type { TrainingRunSummary } from '@public/store/types'
import { STATUS_COLORS } from './constants'

function formatHyperparamValue(value: unknown): string {
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : 'None'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  return String(value)
}

function ComparisonRow({ run }: { run: TrainingRunSummary }) {
  const { data } = useTrainingRunDetail(run.id, true)
  const hyperparameters = (data?.run.hyperparameters ?? null) as Record<string, unknown> | null

  return (
    <Table.Tr>
      <Table.Td>
        <Text size="sm" fw={600}>
          {run.name}
        </Text>
      </Table.Td>
      <Table.Td>
        <Badge variant="light" color={STATUS_COLORS[run.status] ?? 'gray'} tt="capitalize" size="sm">
          {run.status}
        </Badge>
      </Table.Td>
      <Table.Td>
        <Text size="sm">{run.evaluation?.accuracy != null ? run.evaluation.accuracy.toFixed(3) : '—'}</Text>
      </Table.Td>
      <Table.Td>
        <Text size="sm">{run.evaluation?.macroF1 != null ? run.evaluation.macroF1.toFixed(3) : '—'}</Text>
      </Table.Td>
      <Table.Td>
        <Text size="xs" c="dimmed" maw={280}>
          {hyperparameters
            ? Object.entries(hyperparameters)
                .filter(([key]) => key !== 'encoderId')
                .map(([key, value]) => `${key}: ${formatHyperparamValue(value)}`)
                .join(' · ')
            : '—'}
        </Text>
      </Table.Td>
    </Table.Tr>
  )
}

export function RunComparisonPanel({ runs, onBack }: { runs: TrainingRunSummary[]; onBack: () => void }) {
  return (
    <Card withBorder p="lg" radius="md">
      <Group justify="space-between" mb="md">
        <Group gap="sm">
          <Button variant="subtle" color="gray" size="xs" leftSection={<ArrowLeftIcon size={14} />} onClick={onBack}>
            Back to runs
          </Button>
          <Title order={5}>Comparing {runs.length} runs</Title>
        </Group>
      </Group>

      <Table.ScrollContainer minWidth={640}>
        <Table verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Name</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Accuracy</Table.Th>
              <Table.Th>Macro F1</Table.Th>
              <Table.Th>Hyperparameters</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {runs.map((run) => (
              <ComparisonRow key={run.id} run={run} />
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Card>
  )
}
