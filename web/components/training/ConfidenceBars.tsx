import { Group, Text } from '@mantine/core'
import { MeterBar } from '@public/components/ui'

/** Ranked class confidences: the top class in cyan, the rest in slate. */
export function ConfidenceBars({ classes }: { classes: { label: string; confidence: number }[] }) {
  return (
    <div className="flex flex-col gap-2">
      {classes.map((c, i) => (
        <div key={c.label}>
          <Group justify="space-between" gap="xs" wrap="nowrap" mb={2}>
            <Text size="sm" fw={i === 0 ? 600 : 400} truncate>
              {c.label}
            </Text>
            <Text size="xs" c="dimmed" className="tnum">
              {(c.confidence * 100).toFixed(1)}%
            </Text>
          </Group>
          <MeterBar value={c.confidence * 100} color={i === 0 ? 'var(--mantine-color-cyan-5)' : '#475569'} height={8} />
        </div>
      ))}
    </div>
  )
}
