/**
 * RunComparisonPanel – compare a handful of runs: an overlay of one metric across runs, and a table
 * of status, accuracy/macro-F1 (denormalized onto the run summary from its evaluation report) and
 * the hyperparameters each run started with. Hyperparameters and per-epoch metrics aren't in the
 * run list response, so each selected run fetches its own detail.
 */

import { LineChart } from '@mantine/charts'
import { Badge, Group, MultiSelect, Paper, SegmentedControl, Select, Table, Text } from '@mantine/core'
import { EmptyState, SectionLabel, StatusBadge } from '@public/components/ui'
import { CHART_COLORS } from '@public/lib/palette'
import { trainingRunDetailQueryOptions, useTrainingRunDetail } from '@public/lib/queries'
import type { SplitType, TrainingRunSummary } from '@public/store/types'
import { useQueries } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { formatMetricLabel, STATUS_COLORS } from './constants'

function formatHyperparamValue(value: unknown): string {
  if (Array.isArray(value)) return value.length > 0 ? value.join(', ') : 'None'
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toPrecision(4)
  return String(value)
}

function ComparisonRow({ run, color, isCurrent }: { run: TrainingRunSummary; color: string; isCurrent: boolean }) {
  const { data } = useTrainingRunDetail(run.id, true)
  const hyperparameters = (data?.run.hyperparameters ?? null) as Record<string, unknown> | null
  const best = run.evaluation?.status === 'success' ? run.evaluation : null

  return (
    <Table.Tr>
      <Table.Td>
        <Group gap={6} wrap="nowrap">
          <span style={{ width: 8, height: 8, background: color, flex: 'none' }} />
          <Text size="sm" fw={600} c={color}>
            {run.name}
          </Text>
          {isCurrent && <Badge color="cyan">this run</Badge>}
        </Group>
      </Table.Td>
      <Table.Td>
        <StatusBadge value={run.status} colorMap={STATUS_COLORS} />
      </Table.Td>
      <Table.Td className="tnum">{best?.accuracy != null ? best.accuracy.toFixed(3) : '—'}</Table.Td>
      <Table.Td className="tnum">{best?.macroF1 != null ? best.macroF1.toFixed(3) : '—'}</Table.Td>
      <Table.Td>
        <Text size="xs" c="dimmed" maw={320}>
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

export function RunComparisonPanel({
  runs,
  selectedIds,
  onSelectedChange,
  currentRunId,
}: {
  /** Every run that can be compared. */
  runs: TrainingRunSummary[]
  selectedIds: string[]
  onSelectedChange: (ids: string[]) => void
  currentRunId?: string
}) {
  const [metric, setMetric] = useState('loss')
  const [split, setSplit] = useState<SplitType>('validation')
  const selected = selectedIds.map((id) => runs.find((r) => r.id === id)).filter((r): r is TrainingRunSummary => !!r)

  const details = useQueries({
    queries: selected.map((r) => ({ ...trainingRunDetailQueryOptions(r.id), enabled: true })),
  })

  const metricNames = useMemo(() => {
    const names = new Set<string>()
    for (const d of details) for (const m of d.data?.run.metrics ?? []) names.add(m.metricName)
    return Array.from(names).sort()
  }, [details])
  const activeMetric = metricNames.includes(metric) ? metric : (metricNames[0] ?? metric)

  // One row per epoch; a column per run (keyed by run id) holding the chosen metric on the chosen split.
  const data = useMemo(() => {
    const byEpoch = new Map<number, Record<string, number>>()
    selected.forEach((run, i) => {
      for (const m of details[i]?.data?.run.metrics ?? []) {
        if (m.metricName !== activeMetric || m.split !== split) continue
        const row = byEpoch.get(m.epoch) ?? { epoch: m.epoch }
        row[run.id] = m.metricValue
        byEpoch.set(m.epoch, row)
      }
    })
    return Array.from(byEpoch.values()).sort((a, b) => a.epoch - b.epoch)
  }, [details, selected, activeMetric, split])

  const colorFor = (id: string) => CHART_COLORS[Math.max(0, selectedIds.indexOf(id)) % CHART_COLORS.length]

  return (
    <div className="flex flex-col gap-3">
      <Paper p="sm">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <MultiSelect
            size="xs"
            label="Runs"
            w={360}
            searchable
            data={runs.map((r) => ({ value: r.id, label: r.name }))}
            value={selectedIds}
            onChange={onSelectedChange}
          />
          <Select
            size="xs"
            label="Metric"
            w={200}
            data={metricNames.map((n) => ({ value: n, label: formatMetricLabel(n) }))}
            value={activeMetric}
            onChange={(v) => v && setMetric(v)}
            allowDeselect={false}
            disabled={metricNames.length === 0}
          />
          <SegmentedControl
            value={split}
            onChange={(v) => setSplit(v as SplitType)}
            data={[
              { value: 'train', label: 'Train' },
              { value: 'validation', label: 'Validation' },
              { value: 'test', label: 'Test' },
            ]}
          />
        </Group>
      </Paper>

      {selected.length === 0 ? (
        <EmptyState title="Pick runs to compare" description="Select two or more runs above." compact />
      ) : (
        <Paper p="sm">
          <SectionLabel mb={4}>
            {formatMetricLabel(activeMetric)} · {split}
          </SectionLabel>
          <LineChart
            h={280}
            data={data}
            dataKey="epoch"
            series={selected.map((r) => ({
              name: r.id,
              label: r.name,
              color: colorFor(r.id),
              strokeWidth: r.id === currentRunId ? 3 : 1.75,
            }))}
            withLegend
            withDots={false}
            curveType="monotone"
            connectNulls
            gridAxis="y"
          />
        </Paper>
      )}

      {selected.length > 0 && (
        <Paper>
          <Table.ScrollContainer minWidth={640}>
            <Table verticalSpacing={6}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Run</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Accuracy</Table.Th>
                  <Table.Th>Macro F1</Table.Th>
                  <Table.Th>Hyperparameters</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {selected.map((run) => (
                  <ComparisonRow key={run.id} run={run} color={colorFor(run.id)} isCurrent={run.id === currentRunId} />
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Paper>
      )}
    </div>
  )
}
