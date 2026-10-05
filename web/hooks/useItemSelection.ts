import type { ItemSelection } from '@public/components/dataset/items/types'
import {
  emptySelection,
  isSelected,
  type MatchFilter,
  replaceSelection,
  type SelectionState,
  selectAllMatching,
  selectedCount,
  selectRange,
  setMany,
  toggleItem,
  toRequest,
} from '@public/lib/itemSelection'
import { useEffect, useState } from 'react'

/**
 * Checkbox selection for the item browser. `visibleIds` is the current page in
 * display order (what shift-click ranges over); `resetKey` clears the selection
 * when the filter that produced it changes, since a selection made under one
 * filter would otherwise silently apply to rows the user can no longer see.
 * Changing page does not reset it, so a selection can span pages.
 */
export function useItemSelection({ visibleIds, resetKey }: { visibleIds: readonly string[]; resetKey: string }) {
  const [state, setState] = useState<SelectionState>(emptySelection)

  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional reset-on-change, resetKey isn't read in the body
  useEffect(() => setState(emptySelection), [resetKey])

  const selection: ItemSelection = {
    isSelected: (itemId) => isSelected(state, itemId),
    onToggle: (itemId, range) =>
      setState((prev) => (range ? selectRange(prev, visibleIds, itemId) : toggleItem(prev, itemId))),
    onToggleMany: (itemIds, selected) => setState((prev) => setMany(prev, itemIds, selected)),
    onClear: () => setState(emptySelection),
  }

  return {
    selection,
    /** True after "select all matching": the selection then follows the filter, not a list of ids. */
    allMatching: state.all,
    /** Items left out of an "all matching" selection (the server accepts a limited number). */
    exclusionCount: state.all ? state.ids.size : 0,
    count: (matching: number) => selectedCount(state, matching),
    /** The `itemIds` / `filter` half of a bulk request body. */
    request: (filter: MatchFilter) => toRequest(state, filter),
    clear: () => setState(emptySelection),
    selectPage: (ids: readonly string[]) => setState(replaceSelection(ids)),
    selectAllMatching: () => setState(selectAllMatching()),
  }
}
