import { QueryBoundary } from '@public/components/ui'
import type { Modality } from '@public/store/types'
import type { ReactNode } from 'react'
import { ItemsTable } from './items/ItemsTable'
import { MediaGrid } from './items/MediaGrid'
import type { CaptionEditing, DatasetListItem, ItemSelection } from './items/types'

export type { CaptionEditing, DatasetListItem, ItemSelection } from './items/types'

/**
 * Dispatches to a thumbnail grid (vision/audio — click a card to open a
 * modal with the full image/audio player and metadata) or a table
 * (text/tabular), by dataset modality. Shared by the Dataset page's draft
 * browser and the Snapshot detail page's per-version/split item viewer so both
 * present items the same way. Never shows the item's internal id — only
 * externalId, which is what an operator actually recognizes an item by.
 *
 * Pass `selection` to turn on a checkbox per item (grid: card corner, table:
 * leading column with a header "select all this page" checkbox) for pages
 * that need batch operations — the caller owns the selected-id set.
 */
export function ItemsByModality<T extends DatasetListItem>({
  items,
  modality,
  isLoading,
  isError,
  onRetry,
  classNameById,
  emptyMessage = 'No items yet',
  renderActions,
  selection,
  captionEditing,
}: {
  items: T[]
  modality: Modality
  isLoading?: boolean
  isError?: boolean
  onRetry?: () => void
  classNameById: Map<string, string>
  emptyMessage?: string
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  /** Enables per-item free-text caption/transcript editing — vision/audio modalities only (see `CaptionEditor`). */
  captionEditing?: CaptionEditing
}) {
  // A failed fetch must not fall through to `emptyMessage` — "No items in the
  // draft yet" is a very different claim from "the request failed".
  if (isError) return <QueryBoundary isLoading={false} isError onRetry={onRetry} compact />

  if (modality === 'vision' || modality === 'audio') {
    return (
      <MediaGrid
        items={items}
        modality={modality}
        isLoading={isLoading}
        classNameById={classNameById}
        emptyMessage={emptyMessage}
        renderActions={renderActions}
        captionEditing={captionEditing}
        selection={selection}
      />
    )
  }

  return (
    <ItemsTable
      items={items}
      modality={modality}
      isLoading={isLoading}
      classNameById={classNameById}
      emptyMessage={emptyMessage}
      renderActions={renderActions}
      selection={selection}
    />
  )
}
