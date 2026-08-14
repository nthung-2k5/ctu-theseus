import { Text } from '@mantine/core'
import { modals } from '@mantine/modals'
import type { ReactNode } from 'react'

/**
 * Opens Mantine's confirm modal with the delete-flavored defaults every
 * page currently hand-rolls (red confirm button, "Are you sure..." copy).
 */
export function confirmDelete({
  title,
  message,
  onConfirm,
}: {
  title: string
  message: ReactNode
  onConfirm: () => void
}) {
  modals.openConfirmModal({
    title,
    children: <Text size="sm">{message}</Text>,
    labels: { confirm: 'Delete', cancel: 'Cancel' },
    confirmProps: { color: 'red' },
    onConfirm,
  })
}
