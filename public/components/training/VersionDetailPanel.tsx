import { LineChart } from '@mantine/charts'
import {
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Progress,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { StopIcon, XCircleIcon } from '@phosphor-icons/react'
import {
  AUGMENTATION_OPTIONS,
  getFamilyName,
  getVariantLabel,
  PREPROCESSING_OPTIONS,
  STATUS_COLORS,
  type TrainingRunData,
} from './constants'

interface VersionDetailPanelProps {
  run: TrainingRunData
  onStop?: () => void
}

export function VersionDetailPanel({ run, onStop }: VersionDetailPanelProps) {
  const lastMetric = run.metrics[run.metrics.length - 1]

  const chartData = run.metrics.map((m) => ({
    epoch: m.epoch,
    'Train Loss': m.trainLoss,
    'Val Loss': m.valLoss,
    Accuracy: m.accuracy,
    mAP: m.mAP,
  }))

  return (
    <Stack gap="lg">
      {/* ── Header ── */}
      <Group justify="space-between" align="flex-start">
        <div>
          <Group gap="sm">
            <Title order={3}>{run.name}</Title>
            <Badge color={STATUS_COLORS[run.status]} variant="light" size="lg" tt="capitalize">
              {run.status}
            </Badge>
          </Group>
          <Text size="sm" c="dimmed" mt={4}>
            {getFamilyName(run.modelId)} · {getVariantLabel(run.modelId, run.variantId)}
          </Text>
        </div>
        {run.status === 'training' && onStop && (
          <Button color="red" variant="light" leftSection={<StopIcon size={16} />} onClick={onStop}>
            Stop Training
          </Button>
        )}
      </Group>

      {/* ── Error banner ── */}
      {run.status === 'failed' && run.error && (
        <Card withBorder p="md" radius="md" bg="var(--mantine-color-red-light)">
          <Group gap="sm">
            <ThemeIcon color="red" variant="filled" size="sm">
              <XCircleIcon size={14} />
            </ThemeIcon>
            <Text size="sm" fw={500} c="red">
              {run.error}
            </Text>
          </Group>
        </Card>
      )}

      {/* ── Queued state ── */}
      {run.status === 'queued' && (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <Loader size="lg" />
            <Text fw={500}>Waiting in queue...</Text>
            <Text size="sm" c="dimmed">
              Training will begin shortly
            </Text>
          </Stack>
        </Card>
      )}

      {/* ── Training progress bar ── */}
      {run.status === 'training' && lastMetric && (
        <Card withBorder p="md" radius="md">
          <Group justify="space-between" mb="xs">
            <Text size="sm" fw={500}>
              Training Progress
            </Text>
            <Text size="sm" c="dimmed">
              Epoch {lastMetric.epoch} / {run.epochs}
            </Text>
          </Group>
          <Progress value={(lastMetric.epoch / run.epochs) * 100} size="md" radius="md" animated color="blue" />
        </Card>
      )}

      {/* ── Summary stats ── */}
      {lastMetric && (
        <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="sm">
          {[
            { label: 'Epoch', value: `${lastMetric.epoch} / ${run.epochs}` },
            { label: 'Train Loss', value: lastMetric.trainLoss.toFixed(4) },
            { label: 'Val Loss', value: lastMetric.valLoss.toFixed(4) },
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
      )}

      {/* ── Charts ── */}
      {run.metrics.length > 0 && (
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

      {/* ── Waiting for metrics ── */}
      {run.status === 'training' && run.metrics.length === 0 && (
        <Card withBorder p="xl" radius="md" ta="center">
          <Stack align="center" gap="sm">
            <Loader />
            <Text size="sm" c="dimmed">
              Waiting for training metrics...
            </Text>
          </Stack>
        </Card>
      )}

      {/* ── Configuration summary ── */}
      <Card withBorder p="md" radius="md">
        <Title order={6} mb="sm">
          Configuration
        </Title>
        <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="xs" mb="md">
          <div>
            <Text size="xs" c="dimmed">
              Architecture
            </Text>
            <Text size="sm" fw={500}>
              {getFamilyName(run.modelId)}
            </Text>
          </div>
          <div>
            <Text size="xs" c="dimmed">
              Variant
            </Text>
            <Text size="sm" fw={500}>
              {getVariantLabel(run.modelId, run.variantId)}
            </Text>
          </div>
          <div>
            <Text size="xs" c="dimmed">
              Epochs
            </Text>
            <Text size="sm" fw={500}>
              {run.epochs}
            </Text>
          </div>
          <div>
            <Text size="xs" c="dimmed">
              Batch Size
            </Text>
            <Text size="sm" fw={500}>
              {run.batchSize}
            </Text>
          </div>
          <div>
            <Text size="xs" c="dimmed">
              Learning Rate
            </Text>
            <Text size="sm" fw={500}>
              {run.learningRate}
            </Text>
          </div>
          <div>
            <Text size="xs" c="dimmed">
              Image Size
            </Text>
            <Text size="sm" fw={500}>
              {run.imageSize}px
            </Text>
          </div>
        </SimpleGrid>

        {/* Preprocessing badges */}
        {Object.entries(run.preprocessing).some(([, v]) => v.enabled) && (
          <>
            <Text size="xs" c="dimmed" mb={4}>
              Preprocessing
            </Text>
            <Group gap="xs" mb="sm">
              {Object.entries(run.preprocessing)
                .filter(([, v]) => v.enabled)
                .map(([key]) => {
                  const opt = PREPROCESSING_OPTIONS.find((o) => o.id === key)
                  return (
                    <Badge key={key} variant="light" size="sm">
                      {opt?.name ?? key}
                    </Badge>
                  )
                })}
            </Group>
          </>
        )}

        {/* Augmentation badges */}
        {Object.entries(run.augmentation).some(([, v]) => v.enabled) && (
          <>
            <Text size="xs" c="dimmed" mb={4}>
              Augmentation
            </Text>
            <Group gap="xs">
              {Object.entries(run.augmentation)
                .filter(([, v]) => v.enabled)
                .map(([key]) => {
                  const opt = AUGMENTATION_OPTIONS.find((o) => o.id === key)
                  return (
                    <Badge key={key} variant="light" color="violet" size="sm">
                      {opt?.name ?? key}
                    </Badge>
                  )
                })}
            </Group>
          </>
        )}
      </Card>
    </Stack>
  )
}
