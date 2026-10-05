import { QueryBoundary } from '@public/components/ui'
import type { Modality } from '@public/store/types'
import { type ReactNode, useState } from 'react'
import { CLASS_GROUND_TRUTH, type GroundTruthSpec } from './items/groundTruth'
import { ItemActionsMenu } from './items/ItemActionsMenu'
import { ItemDrawer } from './items/ItemDrawer'
import { ItemsTable } from './items/ItemsTable'
import { MediaGrid } from './items/MediaGrid'
import type { CaptionEditing, DatasetListItem, ItemEditing, ItemSelection } from './items/types'

export type { GroundTruthSpec } from './items/groundTruth'
export type { CaptionEditing, DatasetListItem, ItemEditing, ItemSelection } from './items/types'

/**
 * Dispatches to a thumbnail grid (vision/audio) or a table (text/tabular) by
 * dataset modality; clicking an item opens it in a drawer that steps through
 * the page. Shared by the Dataset page's draft browser and the Snapshot detail
 * page's per-version/split item viewer so both present items the same way.
 * Never shows the item's internal id — only externalId, which is what an
 * operator actually recognizes an item by.
 *
 * `annotation` is the task registry's ground-truth spec and decides how a label
 * is shown and edited (class, numeric target, free text, token tags).
 *
 * Pass `selection` to turn on a checkbox per item (grid: card corner, table:
 * leading column with a header "select all this page" checkbox) for pages
 * that need batch operations — the caller owns the selected-id set.
 *
 * Pass `editing` to let the drawer and a per-item menu change an item (split,
 * class, target, delete); leave it out for an immutable view such as a snapshot.
 */
export function ItemsByModality<T extends DatasetListItem>({
  items,
  modality,
  annotation = CLASS_GROUND_TRUTH,
  isLoading,
  isError,
  onRetry,
  classNameById,
  emptyMessage = 'No items yet',
  renderActions,
  selection,
  captionEditing,
  editing,
}: {
  items: T[]
  modality: Modality
  annotation?: GroundTruthSpec
  isLoading?: boolean
  isError?: boolean
  onRetry?: () => void
  classNameById: Map<string, string>
  emptyMessage?: string
  /** Overrides the default per-item menu. */
  renderActions?: (item: T) => ReactNode
  selection?: ItemSelection
  /** Enables per-item free-text editing of the ground truth (see `CaptionEditor`). */
  captionEditing?: CaptionEditing
  editing?: ItemEditing
}) {
  // Held by id and looked up in the live list, so an edit shows up in the drawer after the refetch,
  // and an item that leaves the list (deleted, or edited out of the current filter) closes it.
  const [openId, setOpenId] = useState<string | null>(null)
  const openIndex = openId === null ? -1 : items.findIndex((i) => i.id === openId)
  const openItem = openIndex === -1 ? null : items[openIndex]

  // A failed fetch must not fall through to `emptyMessage` — "No items in the
  // draft yet" is a very different claim from "the request failed".
  if (isError) return <QueryBoundary isLoading={false} isError onRetry={onRetry} compact />

  const onOpen = (item: T) => setOpenId(item.id)
  const rowActions =
    renderActions ??
    (editing ? (item: T) => <ItemActionsMenu item={item} editing={editing} onOpen={() => onOpen(item)} /> : undefined)

  const list =
    modality === 'vision' || modality === 'audio' ? (
      <MediaGrid
        items={items}
        modality={modality}
        isLoading={isLoading}
        spec={annotation}
        classNameById={classNameById}
        emptyMessage={emptyMessage}
        onOpen={onOpen}
        renderActions={rowActions}
        captionEditing={captionEditing}
        selection={selection}
      />
    ) : (
      <ItemsTable
        items={items}
        modality={modality}
        isLoading={isLoading}
        spec={annotation}
        classNameById={classNameById}
        emptyMessage={emptyMessage}
        onOpen={onOpen}
        renderActions={rowActions}
        selection={selection}
        captionEditing={captionEditing}
      />
    )

  return (
    <>
      {list}
      <ItemDrawer
        item={openItem}
        position={openIndex + 1}
        total={items.length}
        onPrev={openIndex > 0 ? () => setOpenId(items[openIndex - 1].id) : undefined}
        onNext={openIndex !== -1 && openIndex < items.length - 1 ? () => setOpenId(items[openIndex + 1].id) : undefined}
        onClose={() => setOpenId(null)}
        modality={modality}
        spec={annotation}
        classNameById={classNameById}
        captionEditing={captionEditing}
        editing={editing}
      />
    </>
  )
}
