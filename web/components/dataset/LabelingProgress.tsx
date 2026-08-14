import { Group, Progress, Text } from '@mantine/core'

/** "N / total labeled" indicator shown on DataPage/DatasetPage for tasks that require label classes. */
export function LabelingProgress({ labeled, total }: { labeled: number; total: number }) {
  const pct = total > 0 ? Math.round((labeled / total) * 100) : 0
  return (
    <Group gap="sm" wrap="nowrap">
      <Progress value={pct} w={120} size="sm" color={pct === 100 ? 'teal' : 'primary'} />
      <Text size="xs" c="dimmed">
        {labeled} / {total} labeled
      </Text>
    </Group>
  )
}
