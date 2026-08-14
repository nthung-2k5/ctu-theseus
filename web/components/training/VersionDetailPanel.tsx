/**
 * VersionDetailPanel – live console for a training run.
 *
 * Metrics and logs stream over SSE (useRunEvents) instead of polling; the
 * chart and log ring buffer update in place as events arrive, and survive a
 * hard refresh via the browser's native Last-Event-ID replay.
 */

import { LineChart } from '@mantine/charts'
import { Badge, Button, Card, Code, Group, Loader, ScrollArea, Stack, Table, Text, Title } from '@mantine/core'
import { StopIcon } from '@phosphor-icons/react'
import { useRunEvents } from '@public/hooks/useRunEvents'
import { trainingRunDetailQueryOptions, trainingRunsQueryOptions, useTrainingRunDetail } from '@public/lib/queries'
import type { TrainingMetric, TrainingRunSummary } from '@public/store/types'
import { useQueryClient } from '@tanstack/react-query'
import { STATUS_COLORS } from './constants'

interface VersionDetailPanelProps {
  projectId: string
  run: TrainingRunSummary
  onStop?: () => void
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

  const series = Array.from(new Set(live.metricPoints.flatMap((p) => Object.keys(p).filter((k) => k !== 'epoch')))).map(
    (key, i) => ({
      name: key,
      color: ['blue.6', 'teal.6', 'orange.6', 'grape.6', 'red.6', 'green.6'][i % 6],
    }),
  )

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
              <Button size="xs" color="red" variant="light" leftSection={<StopIcon size={14} />} onClick={onStop}>
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

        {/* Live metric chart */}
        {live.metricPoints.length > 0 && (
          <div>
            <Text size="sm" fw={600} mb="xs">
              Metrics
            </Text>
            <LineChart
              h={260}
              data={live.metricPoints}
              dataKey="epoch"
              series={series}
              withLegend
              curveType="monotone"
            />
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

              {finishedMetrics.length > 0 && (
                <div>
                  <Text size="sm" fw={600} mb="xs">
                    Final Metrics
                  </Text>
                  <Table striped highlightOnHover>
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Epoch</Table.Th>
                        <Table.Th>Split</Table.Th>
                        <Table.Th>Metric</Table.Th>
                        <Table.Th>Value</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {finishedMetrics.map((m: TrainingMetric) => (
                        <Table.Tr key={`${m.epoch}-${m.split}-${m.metricName}`}>
                          <Table.Td>{m.epoch}</Table.Td>
                          <Table.Td tt="capitalize">{m.split}</Table.Td>
                          <Table.Td>{m.metricName}</Table.Td>
                          <Table.Td>{m.metricValue.toFixed(4)}</Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </div>
              )}
            </>
          ) : null)}
      </Stack>
    </Card>
  )
}
