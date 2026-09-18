/**
 * Dataset health / EDA panel — surfaces stats that already exist in
 * visionFeatures/audioFeatures/textFeatures/labelClasses but nothing in the
 * UI showed before: class balance, content-hash integrity, and per-modality
 * distributions. Scoped to the draft (what a snapshot cut right now would
 * actually contain), not the full historical pool.
 *
 * See server/routes/datasets.ts's GET /projects/:projectId/dataset/health.
 */

import { Alert, Badge, Group, Progress, SimpleGrid, Skeleton, Stack, Text, Title } from '@mantine/core'
import {
  CheckCircleIcon,
  FileImageIcon,
  HashIcon,
  SpeakerHighIcon,
  TableIcon,
  TagIcon,
  TextTIcon,
  WarningCircleIcon,
} from '@phosphor-icons/react'
import { EmptyState, StatCard } from '@public/components/ui'
import { useDatasetHealth } from '@public/lib/queries'
import type { DatasetHealthReport } from '@public/store/types'

/** Deterministic color per class name so the same class keeps its bar color across renders. */
const BAR_COLORS = ['blue', 'teal', 'grape', 'orange', 'cyan', 'pink', 'lime', 'indigo', 'red', 'yellow']
function colorForIndex(i: number): string {
  return BAR_COLORS[i % BAR_COLORS.length]
}

function ClassDistribution({ report }: { report: DatasetHealthReport }) {
  if (report.classDistribution.length === 0) return null
  const maxCount = Math.max(...report.classDistribution.map((c) => c.count))

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <Text size="sm" fw={600}>
          Class distribution
        </Text>
        <Text size="xs" c="dimmed">
          {report.labeledCount} labeled · {report.unlabeledCount} unlabeled
        </Text>
      </Group>
      <Stack gap={6}>
        {report.classDistribution.map((c, i) => (
          <Group key={c.classId} gap="xs" wrap="nowrap">
            <Text size="xs" w={140} truncate title={c.name}>
              {c.name}
            </Text>
            <Progress.Root size={16} radius="sm" style={{ flex: 1 }}>
              <Progress.Section value={(c.count / maxCount) * 100} color={colorForIndex(i)} />
            </Progress.Root>
            <Text size="xs" c="dimmed" w={40} ta="right">
              {c.count}
            </Text>
          </Group>
        ))}
      </Stack>
      {report.smallClasses.length > 0 && (
        <Alert icon={<WarningCircleIcon size={16} />} color="yellow" variant="light">
          {report.smallClasses.map((c) => c.name).join(', ')} — fewer than 3 items each. These classes can't appear
          in all three splits and will make per-class metrics unreliable.
        </Alert>
      )}
    </Stack>
  )
}

function IntegritySection({ report }: { report: DatasetHealthReport }) {
  const hasIssues = report.duplicateContentHashes > 0 || report.missingContentHash > 0
  if (!hasIssues) {
    return (
      <Alert icon={<CheckCircleIcon size={16} />} color="green" variant="light">
        No duplicate content found — uploads are deduplicated by content hash automatically.
      </Alert>
    )
  }
  return (
    <Alert icon={<WarningCircleIcon size={16} />} color="yellow" variant="light">
      {report.duplicateContentHashes > 0 &&
        `${report.duplicateContentHashes} content hash(es) appear on more than one item. `}
      {report.missingContentHash > 0 &&
        `${report.missingContentHash} item(s) have no content hash and are exempt from automatic dedup.`}
    </Alert>
  )
}

function MinMaxAvgRow({
  label,
  unit,
  stats,
}: {
  label: string
  unit: string
  stats: { min: number; max: number; avg: number }
}) {
  return (
    <Group gap="xs">
      <Text size="xs" c="dimmed" w={90}>
        {label}
      </Text>
      <Text size="xs">
        min {stats.min}
        {unit} · avg {stats.avg}
        {unit} · max {stats.max}
        {unit}
      </Text>
    </Group>
  )
}

function BreakdownChips({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort(([, a], [, b]) => b - a)
  return (
    <Group gap={6}>
      {entries.map(([key, value]) => (
        <Badge key={key} variant="light" color="gray">
          {key}: {value}
        </Badge>
      ))}
    </Group>
  )
}

function ModalitySection({ report }: { report: DatasetHealthReport }) {
  if (report.vision) {
    return (
      <Stack gap="xs">
        <Group gap="xs">
          <FileImageIcon size={16} />
          <Text size="sm" fw={600}>
            Image dimensions
          </Text>
        </Group>
        <MinMaxAvgRow label="Width" unit="px" stats={report.vision.width} />
        <MinMaxAvgRow label="Height" unit="px" stats={report.vision.height} />
        <BreakdownChips counts={report.vision.formats} />
      </Stack>
    )
  }
  if (report.audio) {
    return (
      <Stack gap="xs">
        <Group gap="xs">
          <SpeakerHighIcon size={16} />
          <Text size="sm" fw={600}>
            Audio duration
          </Text>
        </Group>
        <MinMaxAvgRow label="Duration" unit="s" stats={report.audio.durationSeconds} />
        <BreakdownChips counts={report.audio.sampleRates} />
      </Stack>
    )
  }
  if (report.text) {
    return (
      <Stack gap="xs">
        <Group gap="xs">
          <TextTIcon size={16} />
          <Text size="sm" fw={600}>
            Text length
          </Text>
        </Group>
        {report.text.tokenCount ? (
          <MinMaxAvgRow label="Tokens" unit="" stats={report.text.tokenCount} />
        ) : (
          <Text size="xs" c="dimmed">
            No token counts recorded for these items.
          </Text>
        )}
        <BreakdownChips counts={report.text.languages} />
      </Stack>
    )
  }
  if (report.tabular) {
    return (
      <Group gap="xs">
        <TableIcon size={16} />
        <Text size="sm" c="dimmed">
          {report.tabular.count} tabular item{report.tabular.count === 1 ? '' : 's'} — per-field stats aren't
          available yet since features are stored as free-form JSON.
        </Text>
      </Group>
    )
  }
  return null
}

export function DatasetHealthPanel({ projectId, active }: { projectId: string; active: boolean }) {
  const { data, isLoading } = useDatasetHealth(projectId, active)

  if (isLoading) {
    return (
      <Stack gap="md">
        <Skeleton height={80} />
        <Skeleton height={120} />
      </Stack>
    )
  }

  const report = data?.health as DatasetHealthReport | undefined
  if (!report) return null

  if (report.itemCount === 0) {
    return (
      <EmptyState
        icon={HashIcon}
        title="Nothing to analyze yet"
        description="Add items to the draft to see class balance and modality stats."
      />
    )
  }

  return (
    <Stack gap="lg">
      <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="md">
        <StatCard icon={HashIcon} label="Items" value={report.itemCount} compact />
        <StatCard icon={TagIcon} label="Labeled" value={report.labeledCount} compact />
        <StatCard icon={WarningCircleIcon} label="Unlabeled" value={report.unlabeledCount} compact color="gray" />
      </SimpleGrid>

      <div>
        <Title order={6} mb="xs">
          Class balance
        </Title>
        {report.classDistribution.length > 0 ? (
          <ClassDistribution report={report} />
        ) : (
          <Text size="xs" c="dimmed">
            This task doesn't use label classes, or none have been assigned yet.
          </Text>
        )}
      </div>

      <div>
        <Title order={6} mb="xs">
          Data integrity
        </Title>
        <IntegritySection report={report} />
      </div>

      <div>
        <Title order={6} mb="xs">
          Modality stats
        </Title>
        <ModalitySection report={report} />
      </div>
    </Stack>
  )
}
