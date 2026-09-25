import { Group, Paper, Text, ThemeIcon } from '@mantine/core'
import type { Icon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'
import { SectionLabel } from './SectionLabel'

/**
 * KPI tile: uppercase label, tabular value and an optional hint. The icon is optional and
 * rendered small.
 */
export function StatCard({
  icon: TheIcon,
  color = 'primary',
  label,
  value,
  hint,
}: {
  icon?: Icon
  color?: string
  label: string
  value: ReactNode
  hint?: ReactNode
  /** @deprecated all tiles are dense now; accepted so existing call sites keep compiling. */
  compact?: boolean
}) {
  return (
    <Paper p="sm">
      <Group gap="xs" wrap="nowrap" align="flex-start">
        {TheIcon && (
          <ThemeIcon size="md" variant="light" color={color}>
            <TheIcon size={16} />
          </ThemeIcon>
        )}
        <div style={{ minWidth: 0 }}>
          <SectionLabel>{label}</SectionLabel>
          <Text fw={600} size="xl" className="tnum">
            {value}
          </Text>
          {hint && (
            <Text size="xs" c="dimmed">
              {hint}
            </Text>
          )}
        </div>
      </Group>
    </Paper>
  )
}
