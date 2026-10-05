import type { ItemSelection } from '@public/components/dataset/items/types'
import {
  emptySelection,
  replaceSelection,
  type SelectionState,
  selectRange,
  setMany,
  toggleItem,
} from '@public/lib/itemSelection'
import { useEffect, useState } from 'react'

/**
 * Checkbox selection for the item browser. `visibleIds` is the current page in
 * display order (what shift-click ranges over); `resetKey` clears the selection
 * when the filter that produced it changes, since a selection made under one
 * filter would otherwise silently apply to rows the user can no longer see.
 */
export function useItemSelection({ visibleIds, resetKey }: { visibleIds: readonly string[]; resetKey: string }) {
  const [state, setState] = useState<SelectionState>(emptySelection)

  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional reset-on-change, resetKey isn't read in the body
  useEffect(() => setState(emptySelection), [resetKey])

  const selection: ItemSelection = {
    selectedIds: state.ids,
    onToggle: (itemId, range) =>
      setState((prev) => (range ? selectRange(prev, visibleIds, itemId) : toggleItem(prev, itemId))),
    onToggleMany: (itemIds, selected) => setState((prev) => setMany(prev, itemIds, selected)),
  }

  return {
    selection,
    selectedIds: state.ids,
    clear: () => setState(emptySelection),
    replace: (ids: readonly string[]) => setState(replaceSelection(ids)),
  }
}
