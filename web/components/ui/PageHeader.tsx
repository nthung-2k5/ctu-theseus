import { Group, Text, Title } from '@mantine/core'
import type { ReactNode } from 'react'

/**
 * Standard page title + subtitle + optional badges and right-aligned actions, used at the top of
 * every page body. Dense: an order-4 title with a dimmed xs subtitle.
 */
export function PageHeader({
  title,
  description,
  badges,
  actions,
}: {
  title: ReactNode
  description?: ReactNode
  badges?: ReactNode
  actions?: ReactNode
}) {
  return (
    <Group justify="space-between" align="flex-start" wrap="nowrap">
      <div style={{ minWidth: 0 }}>
        <Group gap="xs" align="center">
          <Title order={4}>{title}</Title>
          {badges}
        </Group>
        {description && (
          <Text size="xs" c="dimmed" mt={2}>
            {description}
          </Text>
        )}
      </div>
      {actions && (
        <Group gap="xs" wrap="nowrap">
          {actions}
        </Group>
      )}
    </Group>
  )
}
