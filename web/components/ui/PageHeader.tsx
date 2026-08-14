import { Group, Text, Title } from '@mantine/core'
import type { ReactNode } from 'react'

/** Standard page title + description + optional right-aligned actions, used at the top of every page body. */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <Group justify="space-between" align="flex-start">
      <div>
        <Title order={2}>{title}</Title>
        {description && (
          <Text size="sm" c="dimmed" mt={4}>
            {description}
          </Text>
        )}
      </div>
      {actions}
    </Group>
  )
}
