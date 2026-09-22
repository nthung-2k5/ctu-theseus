/**
 * VersionDetailPanel – live console for a training run.
 *
 * Metrics and logs stream over SSE (useRunEvents) instead of polling; the
 * chart and log ring buffer update in place as events arrive, and survive a
 * hard refresh via the browser's native Last-Event-ID replay.
 *
 * The metrics section (chart + table) reads from whichever source is live
 * for the run's current state: `live.metricPoints` while the run is active,
 * or the persisted `TrainingMetric` rows (which cover the full epoch
 * history, not just the final epoch) once it has finished. Both are
 * normalized into a common `FlatMetric[]` shape so the chart/table code
 * doesn't need to branch on run status.
 */

import { LineChart } from '@mantine/charts'
import {
  Badge,
  Button,
  Card,
  Chip,
  Code,
  Group,
  Loader,
  ScrollArea,
  SegmentedControl,
  Select,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { DownloadSimpleIcon, StopIcon } from '@phosphor-icons/react'
import { useRunEvents } from '@public/hooks/useRunEvents'
import { trainingRunDetailQueryOptions, trainingRunsQueryOptions, useTrainingRunDetail } from '@public/lib/queries'
import type { SplitType, TrainingRunSummary } from '@public/store/types'
import { useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { formatMetricLabel, SPLIT_COLORS, STATUS_COLORS } from './constants'

interface VersionDetailPanelProps {
  projectId: string
  run: TrainingRunSummary
  onStop?: () => void
}

interface FlatMetric {
  epoch: number
  split: SplitType
  metricName: string
  value: number
}

const SPLIT_ORDER: SplitType[] = ['train', 'validation', 'test']

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function VersionDetailPanel({ projectId, run, onStop }: VersionDetailPanelProps) {
  const isActive = run.status === 'running' || run.status === 'queued'
  const queryClient = useQueryClient()

  const live = useRunEvents(run.id, isActive, () => {
    queryClient.invalidateQueries({ queryKey: trainingRunsQueryOptions(projectId).queryKey })
    queryClient.invalidateQueries({ queryKey: trainingRunDetailQueryOptions(run.id).queryKey })
  })

  const status = live.status ?? run.status

  /* ── Fetch full run detail (hyperparameters, final metrics) once finished ── */
  const { data: detailData, isLoading: detailLoading } = useTrainingRunDetail(run.id, !isActive)
  const detail = detailData?.run
  const finishedMetrics = detail?.metrics ?? []

  /* ── Normalize whichever metric source applies into a flat, uniform shape ── */
  const flatMetrics: FlatMetric[] = useMemo(() => {
    if (isActive) {
      const out: FlatMetric[] = []
      for (const point of live.metricPoints) {
        for (const [key, value] of Object.entries(point)) {
          if (key === 'epoch') continue
          const dot = key.indexOf('.')
          out.push({ epoch: point.epoch, split: key.slice(0, dot) as SplitType, metricName: key.slice(dot + 1), value })
        }
      }
      return out
    }
    return finishedMetrics.map((m) => ({
      epoch: m.epoch,
      split: m.split,
      metricName: m.metricName,
      value: m.metricValue,
    }))
  }, [isActive, live.metricPoints, finishedMetrics])

  const metricIndex = useMemo(() => {
    const idx = new Map<string, number>()
    for (const f of flatMetrics) idx.set(`${f.epoch}|${f.split}|${f.metricName}`, f.value)
    return idx
  }, [flatMetrics])

  const epochs = useMemo(
    () => Array.from(new Set(flatMetrics.map((f) => f.epoch))).sort((a, b) => a - b),
    [flatMetrics],
  )
  const splits = useMemo(() => SPLIT_ORDER.filter((s) => flatMetrics.some((f) => f.split === s)), [flatMetrics])
  const metricNames = useMemo(() => Array.from(new Set(flatMetrics.map((f) => f.metricName))).sort(), [flatMetrics])

  /* ── Chart selection: one metric, one or more splits, x axis = epoch ── */
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)
  const [chartSplits, setChartSplits] = useState<SplitType[]>([])

  const effectiveMetric =
    selectedMetric !== null && metricNames.includes(selectedMetric)
      ? selectedMetric
      : ((metricNames.includes('loss') ? 'loss' : metricNames[0]) ?? null)

  const effectiveChartSplits = chartSplits.filter((s) => splits.includes(s))
  const chartSplitsToUse = effectiveChartSplits.length > 0 ? effectiveChartSplits : splits

  const chartData = useMemo(() => {
    if (!effectiveMetric) return []
    return epochs.map((epoch) => {
      const row: Record<string, number> = { epoch }
      for (const split of chartSplitsToUse) {
        const value = metricIndex.get(`${epoch}|${split}|${effectiveMetric}`)
        if (value !== undefined) row[split] = value
      }
      return row
    })
  }, [epochs, chartSplitsToUse, metricIndex, effectiveMetric])

  const chartSeries = chartSplitsToUse.map((split) => ({
    name: split,
    label: capitalize(split),
    color: SPLIT_COLORS[split] ?? 'blue.6',
  }))

  /* ── Table selection: one epoch, one split, all metrics for that cell ── */
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null)
  const [selectedSplit, setSelectedSplit] = useState<SplitType | null>(null)

  const effectiveEpoch =
    selectedEpoch !== null && epochs.includes(selectedEpoch) ? selectedEpoch : (epochs.at(-1) ?? null)
  const effectiveSplit = selectedSplit !== null && splits.includes(selectedSplit) ? selectedSplit : (splits[0] ?? null)

  const tableRows = useMemo(() => {
    if (effectiveEpoch === null || effectiveSplit === null) return []
    return metricNames
      .filter((name) => metricIndex.has(`${effectiveEpoch}|${effectiveSplit}|${name}`))
      .map((name) => ({ name, value: metricIndex.get(`${effectiveEpoch}|${effectiveSplit}|${name}`)! }))
  }, [metricNames, metricIndex, effectiveEpoch, effectiveSplit])

  return (
    <Card withBorder p="lg" radius="md">
      <Stack gap="lg">
        {/* Header */}
        <Group justify="space-between">
          <div>
            <Title order={4}>{run.name}</Title>
            <Text size="xs" c="dimmed">
              Created: {new Date(run.createdAt).toLocaleString()}
            </Text>
          </div>
          <Group gap="sm">
            <Badge variant="light" color={STATUS_COLORS[status] ?? 'gray'} size="lg" tt="capitalize">
              {isActive && <Loader size={10} mr={4} />}
              {status}
            </Badge>
            {onStop && isActive && (
              <Button
                size="xs"
                color="red"
                variant="light"
                leftSection={<StopIcon weight="fill" size={14} />}
                onClick={onStop}
              >
                Stop
              </Button>
            )}
          </Group>
        </Group>

        {/* Failure message */}
        {live.failedMessage && (
          <Card withBorder p="sm" radius="sm" bg="red.9">
            <Text size="sm" c="red" fw={500}>
              Error
            </Text>
            <Text size="sm">{live.failedMessage}</Text>
          </Card>
        )}

        {/* Metrics: chart + table, both driven by live or persisted metric data */}
        {flatMetrics.length > 0 && effectiveEpoch !== null && effectiveSplit !== null && (
          <div>
            <Text size="sm" fw={600} mb="xs">
              Metrics
            </Text>
            <Stack gap="md">
              <Card withBorder p="md" radius="sm">
                <Group justify="space-between" mb="sm" wrap="wrap">
                  <Select
                    label="Metric"
                    size="xs"
                    w={220}
                    data={metricNames.map((name) => ({ value: name, label: formatMetricLabel(name) }))}
                    value={effectiveMetric}
                    onChange={(v) => setSelectedMetric(v)}
                    allowDeselect={false}
                  />
                  <div>
                    <Text size="xs" fw={500} mb={4}>
                      Splits
                    </Text>
                    <Chip.Group multiple value={chartSplitsToUse} onChange={(v) => setChartSplits(v as SplitType[])}>
                      <Group gap="xs">
                        {splits.map((split) => (
                          <Chip key={split} value={split} size="xs" color={SPLIT_COLORS[split]?.split('.')[0]}>
                            {capitalize(split)}
                          </Chip>
                        ))}
                      </Group>
                    </Chip.Group>
                  </div>
                </Group>
                <LineChart
                  h={260}
                  data={chartData}
                  dataKey="epoch"
                  series={chartSeries}
                  withLegend
                  curveType="monotone"
                />
              </Card>

              <Card withBorder p="md" radius="sm">
                <Group mb="sm" wrap="wrap">
                  <Select
                    label="Epoch"
                    size="xs"
                    w={140}
                    data={epochs.map((e) => ({ value: String(e), label: `Epoch ${e}` }))}
                    value={effectiveEpoch !== null ? String(effectiveEpoch) : null}
                    onChange={(v) => setSelectedEpoch(v !== null ? Number(v) : null)}
                    allowDeselect={false}
                  />
                  <div>
                    <Text size="xs" fw={500} mb={4}>
                      Split
                    </Text>
                    <SegmentedControl
                      size="xs"
                      value={effectiveSplit}
                      onChange={(v) => setSelectedSplit(v as SplitType)}
                      data={splits.map((s) => ({ value: s, label: capitalize(s) }))}
                    />
                  </div>
                </Group>
                <Table striped highlightOnHover>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Metric</Table.Th>
                      <Table.Th>Value</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {tableRows.map((row) => (
                      <Table.Tr key={row.name}>
                        <Table.Td>{formatMetricLabel(row.name)}</Table.Td>
                        <Table.Td>{row.value.toFixed(4)}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </Card>
            </Stack>
          </div>
        )}

        {/* Live log console */}
        {isActive && (
          <div>
            <Group justify="space-between" mb="xs">
              <Text size="sm" fw={600}>
                Console
              </Text>
              <Badge variant="dot" color={live.isConnected ? 'green' : 'gray'} size="xs">
                {live.isConnected ? 'Live' : 'Connecting'}
              </Badge>
            </Group>
            <ScrollArea h={220} bg="dark.8" p="xs" style={{ borderRadius: 6 }}>
              <Code block bg="transparent" style={{ whiteSpace: 'pre-wrap' }}>
                {live.logs.length === 0
                  ? 'Waiting for output…'
                  : live.logs.map((l) => `[${l.level}] ${l.line}`).join('\n')}
              </Code>
            </ScrollArea>
          </div>
        )}

        {/* Run details (after loading) */}
        {!isActive &&
          (detailLoading ? (
            <Card withBorder p="md" radius="sm" ta="center">
              <Loader size="sm" />
            </Card>
          ) : detail ? (
            <>
              {detail.hyperparameters && (
                <div>
                  <Text size="sm" fw={600} mb="xs">
                    Hyperparameters
                  </Text>
                  <Code block>{JSON.stringify(detail.hyperparameters, null, 2)}</Code>
                </div>
              )}

              {detail.failedMessage && !live.failedMessage && (
                <Card withBorder p="sm" radius="sm" bg="red.9">
                  <Text size="sm" c="red" fw={500}>
                    Error
                  </Text>
                  <Text size="sm">{detail.failedMessage}</Text>
                </Card>
              )}

              {/* The live console above only renders while the run is active — the
                  worker uploads the full log regardless, so it's still available here. */}
              <Group>
                <Button
                  size="xs"
                  variant="light"
                  color="gray"
                  leftSection={<DownloadSimpleIcon size={14} />}
                  component="a"
                  href={`/api/runs/${run.id}/logs`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Download logs
                </Button>
              </Group>
            </>
          ) : null)}
      </Stack>
    </Card>
  )
}
