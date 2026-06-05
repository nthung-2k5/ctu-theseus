import { LineChart } from '@mantine/charts'
import { Badge, Button, Card, Group, Loader, Progress, SimpleGrid, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import { StopIcon, XCircleIcon } from '@phosphor-icons/react'
import { useTrainingRunDetail, useTrainingRunStatus } from '@public/queries/training'
import type { TrainingRunSummary } from '@public/store/types'
import { STATUS_COLORS } from './constants'
import { useMemo } from 'react'

interface VersionDetailPanelProps {
  run: TrainingRunSummary
  versionNumber: number
  onStop?: () => void
}

export function VersionDetailPanel({ run, versionNumber, onStop }: VersionDetailPanelProps) {
  const isActive = run.status === 'training' || run.status === 'queued' || run.status === 'preparing'
  const isCompleted = !isActive

  /* ── Poll status for active runs ── */
  const { data: statusData } = useTrainingRunStatus(run.id, isActive)

  /* ── Fetch full detail (with all metrics) for completed runs ── */
  const { data: detailData, isLoading: isDetailLoading } = useTrainingRunDetail(run.id, isCompleted)

  const detail = detailData?.run
  const latestMetrics = statusData?.latestMetrics ?? null
  const isFailed = isCompleted && detail?.failedMessage

  /* ── Chart data from completed run's full metrics ── */
  const chartData = (detail?.trainingMetrics ?? [])
    .sort((a, b) => a.epoch - b.epoch)
    .map((m) => ({
      epoch: m.epoch,
      'Train Loss': m.trainingLoss,
      'Val Loss': m.validationLoss,
      Accuracy: m.accuracy,
      mAP: m.mAP,
    }))

  return (
    <Stack gap="lg">
      {/* ── Header ── */}
      <Group justify="space-between" align="flex-start">
        <div>
          <Group gap="sm">
            <Title order={3}>Version {versionNumber}</Title>
            <Badge color={STATUS_COLORS[run.status]} variant="light" size="lg" tt="capitalize">
              {isActive && <Loader size={10} color={STATUS_COLORS[run.status]} mr={4} />}
              {isFailed ? 'failed' : run.status}
            </Badge>
          </Group>
          <Text size="sm" c="dimmed" mt={4}>
            {run.modelName}
          </Text>
        </div>
        {(run.status === 'training' || run.status === 'queued') && onStop && (
          <Button color="red" variant="light" leftSection={<StopIcon weight="fill" size={16} />} onClick={onStop}>
            Stop Training
          </Button>
        )}
      </Group>

      {/* ── Error banner ── */}
      {isFailed && (
        <Card withBorder p="md" radius="md" bg="var(--mantine-color-red-light)">
          <Group gap="sm">
            <ThemeIcon color="red" variant="filled" size="sm">
              <XCircleIcon size={14} />
            </ThemeIcon>
            <Text size="sm" fw={500} c="red">
              {detail?.failedMessage}
            </Text>
          </Group>
        </Card>
      )}

      {/* ── Queued / Preparing state ── */}
      {(run.status === 'queued' || run.status === 'preparing') && (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <Loader size="lg" />
            <Text fw={500}>{run.status === 'preparing' ? 'Preparing dataset...' : 'Waiting in queue...'}</Text>
            <Text size="sm" c="dimmed">
              Training will begin shortly
            </Text>
          </Stack>
        </Card>
      )}

      {/* ── Training progress (polling latest metrics) ── */}
      {run.status === 'training' && latestMetrics && (
        <Card withBorder p="md" radius="md">
          <Group justify="space-between" mb="xs">
            <Text size="sm" fw={500}>
              Training Progress
            </Text>
            <Text size="sm" c="dimmed">
              Epoch {latestMetrics.epoch}
            </Text>
          </Group>
          <Progress
            value={statusData ? (latestMetrics.epoch / statusData.epochsTotal) * 100 : 0}
            size="md"
            radius="md"
            animated
            color="blue"
          />
        </Card>
      )}

      {/* ── Live summary stats (during training) ── */}
      {run.status === 'training' && latestMetrics && (
        <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="sm">
          {[
            { label: 'Epoch', value: `${latestMetrics.epoch}` },
            { label: 'Train Loss', value: latestMetrics.trainingLoss.toFixed(4) },
            { label: 'Val Loss', value: latestMetrics.validationLoss.toFixed(4) },
            { label: 'Accuracy', value: `${(latestMetrics.accuracy * 100).toFixed(1)}%` },
          ].map((s) => (
            <Card key={s.label} withBorder p="sm" radius="md">
              <Text size="xs" c="dimmed" tt="uppercase">
                {s.label}
              </Text>
              <Text size="lg" fw={700}>
                {s.value}
              </Text>
            </Card>
          ))}
        </SimpleGrid>
      )}

      {/* ── Waiting for first metrics during training ── */}
      {run.status === 'training' && !latestMetrics && (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <Loader />
            <Text size="sm" c="dimmed">
              Waiting for training metrics...
            </Text>
          </Stack>
        </Card>
      )}

      {/* ── Completed run: full summary stats ── */}
      {isCompleted && detail && !isFailed && (
        <>
          {detail.trainingMetrics.length > 0 &&
            (() => {
              const lastMetric = detail.trainingMetrics.sort((a, b) => a.epoch - b.epoch).at(-1)!
              return (
                <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="sm">
                  {[
                    { label: 'Epochs', value: `${lastMetric.epoch} / ${detail.epochs}` },
                    { label: 'Train Loss', value: lastMetric.trainingLoss.toFixed(4) },
                    { label: 'Val Loss', value: lastMetric.validationLoss.toFixed(4) },
                    { label: 'Accuracy', value: `${(lastMetric.accuracy * 100).toFixed(1)}%` },
                  ].map((s) => (
                    <Card key={s.label} withBorder p="sm" radius="md">
                      <Text size="xs" c="dimmed" tt="uppercase">
                        {s.label}
                      </Text>
                      <Text size="lg" fw={700}>
                        {s.value}
                      </Text>
                    </Card>
                  ))}
                </SimpleGrid>
              )
            })()}
        </>
      )}

      {/* ── Loading detail for completed run ── */}
      {isCompleted && isDetailLoading && (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <Loader />
            <Text size="sm" c="dimmed">
              Loading training results...
            </Text>
          </Stack>
        </Card>
      )}

      {/* ── Charts (only for completed runs) ── */}
      {isCompleted && chartData.length > 0 && (
        <>
          <Card withBorder padding="lg" radius="md">
            <Title order={6} mb="md">
              Loss Curves
            </Title>
            <LineChart
              h={250}
              data={chartData}
              dataKey="epoch"
              series={[
                { name: 'Train Loss', color: 'indigo.5' },
                { name: 'Val Loss', color: 'red.5' },
              ]}
              curveType="monotone"
              withDots={false}
            />
          </Card>
          <Card withBorder padding="lg" radius="md">
            <Title order={6} mb="md">
              Accuracy / mAP
            </Title>
            <LineChart
              h={250}
              data={chartData}
              dataKey="epoch"
              series={[
                { name: 'Accuracy', color: 'green.5' },
                { name: 'mAP', color: 'orange.5' },
              ]}
              curveType="monotone"
              withDots={false}
            />
          </Card>
        </>
      )}

      {/* ── Configuration summary (only when detail is loaded) ── */}
      {/* {detail && (
        <Card withBorder p="md" radius="md">
          <Title order={6} mb="sm">
            Configuration
          </Title>
          <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="xs">
            <div>
              <Text size="xs" c="dimmed">
                Model
              </Text>
              <Text size="sm" fw={500}>
                {detail.modelName}
              </Text>
            </div>
            <div>
              <Text size="xs" c="dimmed">
                Epochs
              </Text>
              <Text size="sm" fw={500}>
                {detail.epochs}
              </Text>
            </div>
            <div>
              <Text size="xs" c="dimmed">
                Batch Size
              </Text>
              <Text size="sm" fw={500}>
                {detail.batchSize}
              </Text>
            </div>
            <div>
              <Text size="xs" c="dimmed">
                Learning Rate
              </Text>
              <Text size="sm" fw={500}>
                {detail.learningRate}
              </Text>
            </div>
          </SimpleGrid>
        </Card>
      )} */}
    </Stack>
  )
}
