import { Stack } from '@mantine/core'
import type { ReactNode } from 'react'
import { ItemsPaginationBar, type ItemsPaginationBarProps } from './ItemsPaginationBar'

/**
 * The one sticky region at the bottom of an item list: an optional bulk-action bar above the pager. Stacking
 * them in a single sticky container (rather than two separately-stuck bars at guessed offsets) means they can
 * never overlap each other, and the opaque background keeps cards from showing through the gap between them.
 */
export function ItemsFooter({ bulk, ...pagination }: ItemsPaginationBarProps & { bulk?: ReactNode }) {
  return (
    <Stack
      gap="xs"
      pos="sticky"
      bottom={0}
      py="xs"
      // Above the cards, below drawers and modals.
      style={{ zIndex: 90, background: 'var(--mantine-color-body)' }}
    >
      {bulk}
      <ItemsPaginationBar {...pagination} />
    </Stack>
  )
}
