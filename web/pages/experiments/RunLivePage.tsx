import { LineChart } from '@mantine/charts'
import { Alert, Badge, Group, Loader, Paper, Progress, SimpleGrid, Text } from '@mantine/core'
import { ChartLineIcon } from '@phosphor-icons/react'
import { formatMetricLabel, STATUS_COLORS } from '@public/components/training/constants'
import { LogConsole } from '@public/components/training/LogConsole'
import { useRunContext } from '@public/components/training/RunContext'
import { useRunLogs } from '@public/components/training/useRunLogs'
import { useRunMetrics } from '@public/components/training/useRunMetrics'
import { EmptyState, SectionLabel, StatusBadge } from '@public/components/ui'
import { SERIES_COLORS, VALIDATION_DASH } from '@public/lib/palette'
import { useTrainingRunDetail } from '@public/lib/queries'
import type { SplitType } from '@public/store/types'

const SPLIT_SERIES: Record<SplitType, { color: string; strokeDasharray?: string }> = {
  train: { color: SERIES_COLORS.train },
  validation: { color: SERIES_COLORS.validation, strokeDasharray: VALIDATION_DASH },
  test: { color: SERIES_COLORS.f1 },
}

const fmt = (v: number | null | undefined, digits = 4) => (v == null ? '—' : v.toFixed(digits))

/** Pick up to four metrics to chart: loss first, then accuracy, then whatever else the backend reported. */
function chartedMetrics(names: string[]): string[] {
  const order = ['loss', 'accuracy']
  const first = order.filter((n) => names.includes(n))
  return [...first, ...names.filter((n) => !order.includes(n))].slice(0, 4)
}

export function RunLivePage() {
  const { run, isActive, status, live } = useRunContext()
  const metrics = useRunMetrics(run.id, isActive, live)
  const { data: detail } = useTrainingRunDetail(run.id, true)
  const { lines } = useRunLogs()

  const totalEpochs = Number((detail?.run.hyperparameters as Record<string, unknown> | null)?.epochs) || null
  const currentEpoch = metrics.epochs.at(-1) ?? 0
  const pct = status === 'succeeded' ? 100 : totalEpochs ? Math.min(100, (currentEpoch / totalEpochs) * 100) : 0

  const failure = live.failedMessage ?? detail?.run.failedMessage ?? run.failedMessage
  const shown = chartedMetrics(metrics.metricNames)

  return (
    <div className="flex flex-col gap-3">
      {failure && (
        <Alert color="red" p="xs" title="Run failed">
          {failure}
        </Alert>
      )}

      <Paper p="sm">
        <Group justify="space-between" mb={6}>
          <Group gap="xs">
            <StatusBadge value={status} colorMap={STATUS_COLORS} />
            {isActive && <Loader size={12} />}
            <Text size="xs" c="dimmed" className="tnum">
              {totalEpochs ? `epoch ${currentEpoch} / ${totalEpochs}` : `epoch ${currentEpoch}`}
            </Text>
          </Group>
          {isActive && (
            <Badge variant="dot" color={live.isConnected ? 'teal' : 'gray'}>
              {live.isConnected ? 'streaming' : 'connecting'}
            </Badge>
          )}
        </Group>
        <Progress value={pct} color="cyan" animated={isActive} size="sm" />
      </Paper>

      <SimpleGrid cols={{ base: 2, sm: 3, lg: 5 }} spacing="sm">
        <KpiTile label="Train loss" value={fmt(metrics.latest('train', 'loss'))} color={SERIES_COLORS.train} />
        <KpiTile label="Val loss" value={fmt(metrics.latest('validation', 'loss'))} color={SERIES_COLORS.validation} />
        <KpiTile label="Best val loss" value={fmt(metrics.best('validation', 'loss'))} />
        <KpiTile label="Val accuracy" value={fmt(metrics.latest('validation', 'accuracy'))} color={SERIES_COLORS.f1} />
        <KpiTile label="Epochs done" value={String(currentEpoch)} />
      </SimpleGrid>

      {shown.length === 0 ? (
        <EmptyState
          icon={ChartLineIcon}
          title={isActive ? 'Waiting for the first epoch' : 'No metrics recorded'}
          description={isActive ? 'Curves appear as soon as an epoch completes.' : undefined}
          compact
        />
      ) : (
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
          {shown.map((name) => (
            <Paper key={name} p="sm">
              <SectionLabel mb={4}>{formatMetricLabel(name)}</SectionLabel>
              <LineChart
                h={200}
                data={metrics.series(name)}
                dataKey="epoch"
                series={metrics.splits.map((split) => ({
                  name: split,
                  label: split[0].toUpperCase() + split.slice(1),
                  ...SPLIT_SERIES[split],
                }))}
                withLegend
                withDots={false}
                curveType="monotone"
                connectNulls
                gridAxis="y"
              />
            </Paper>
          ))}
        </SimpleGrid>
      )}

      <Paper style={{ overflow: 'hidden' }}>
        <div style={{ height: 300 }}>
          <LogConsole
            lines={lines}
            empty={isActive ? 'Waiting for output…' : 'No log output was recorded for this run.'}
          />
        </div>
      </Paper>
    </div>
  )
}

function KpiTile({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <Paper p="sm">
      <SectionLabel>{label}</SectionLabel>
      <Text fw={600} size="xl" className="tnum" style={color ? { color } : undefined}>
        {value}
      </Text>
    </Paper>
  )
}
