import { Loader, Paper, Stack, Text, ThemeIcon } from '@mantine/core'
import type { Icon } from '@phosphor-icons/react'
import type { ReactNode } from 'react'

/**
 * The "nothing here yet" panel repeated across every page (empty pool, no snapshots, no runs, ...),
 * and doubling as the loading placeholder when `loading` is set.
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
    <Paper p={compact ? 'md' : 'xl'} ta="center">
      {loading ? (
        <Loader size="sm" />
      ) : (
        <Stack align="center" gap="xs">
          {TheIcon && (
            <ThemeIcon size={compact ? 36 : 44} variant="light" color="gray" radius="xl">
              <TheIcon size={compact ? 18 : 24} weight="thin" />
            </ThemeIcon>
          )}
          {title && (
            <Text fw={500} size="md">
              {title}
            </Text>
          )}
          {description && (
            <Text size="sm" c="dimmed" maw={400}>
              {description}
            </Text>
          )}
          {action}
        </Stack>
      )}
    </Paper>
  )
}
