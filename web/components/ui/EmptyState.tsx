import { Card, Loader, Stack, Text, ThemeIcon, Title } from '@mantine/core'
import type { Icon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

/**
 * The "nothing here yet" card repeated across every page (empty pool,
 * no snapshots, no runs, ...), and doubling as the loading placeholder
 * when `loading` is set — same card shape, just a spinner instead of an
 * icon/title.
 */
export function EmptyState({
  icon: TheIcon,
  title,
  description,
  action,
  compact = false,
  loading = false,
}: {
  icon?: Icon
  title?: string
  description?: ReactNode
  action?: ReactNode
  compact?: boolean
  loading?: boolean
}) {
  return (
    <Card withBorder p={compact ? 'lg' : 'xl'} radius="md" ta="center">
      {loading ? (
        <Loader size="sm" />
      ) : (
        <Stack align="center" gap={compact ? 'sm' : 'md'}>
          {TheIcon && (
            <ThemeIcon size={compact ? 44 : 56} variant="light" color="gray" radius="xl">
              <TheIcon size={compact ? 22 : 30} weight="thin" />
            </ThemeIcon>
          )}
          {title && <Title order={5}>{title}</Title>}
          {description && (
            <Text size="sm" c="dimmed" maw={400}>
              {description}
            </Text>
          )}
          {action}
        </Stack>
      )}
    </Card>
  )
}
