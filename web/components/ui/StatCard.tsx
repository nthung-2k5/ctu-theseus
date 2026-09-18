import { Card, Group, Text, ThemeIcon } from '@mantine/core'
import type { Icon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

/**
 * A labeled stat tile with an icon — the "Pool Items / Classes / Snapshots /
 * Runs" cards repeated across project pages.
 *
 * `compact` is the denser variant used in the project overview's 2-column
 * summary grid.
 */
export function StatCard({
  icon: TheIcon,
  color = 'primary',
  label,
  value,
  compact = false,
}: {
  icon: Icon
  color?: string
  label: string
  value: ReactNode
  compact?: boolean
}) {
  return (
    <Card withBorder padding={compact ? 'sm' : 'lg'} radius="md">
      <Group gap={compact ? 'xs' : undefined} wrap="nowrap">
        <ThemeIcon size={compact ? 'md' : 'lg'} variant="light" color={color} radius="md">
          <TheIcon size={compact ? 18 : 22} />
        </ThemeIcon>
        <div>
          <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
            {label}
          </Text>
          <Text size={compact ? 'lg' : 'xl'} fw={700}>
            {value}
          </Text>
        </div>
      </Group>
    </Card>
  )
}
