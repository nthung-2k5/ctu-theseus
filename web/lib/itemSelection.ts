/**
 * Pure selection-state transitions for the item browser. Kept free of React so
 * they can be unit-tested; `hooks/useItemSelection.ts` wraps them in state.
 *
 * Two modes. Normally `ids` are the selected items. After "select all matching",
 * `all` is true and `ids` are the items *left out* of that selection, so ticking
 * a few boxes off doesn't need every matching id on the client.
 *
 * `anchor` is the last item toggled directly, the fixed end of a shift-click range.
 */
export interface SelectionState {
  ids: ReadonlySet<string>
  anchor: string | null
  all: boolean
}

export const emptySelection: SelectionState = { ids: new Set(), anchor: null, all: false }

export const isSelected = (state: SelectionState, id: string): boolean => state.ids.has(id) !== state.all

/** How many items the selection covers, given how many match the current filter. */
export const selectedCount = (state: SelectionState, matching: number): number =>
  state.all ? Math.max(0, matching - state.ids.size) : state.ids.size

export const toggleItem = (state: SelectionState, id: string): SelectionState => {
  const ids = new Set(state.ids)
  if (ids.has(id)) ids.delete(id)
  else ids.add(id)
  return { ...state, ids, anchor: id }
}

export const setMany = (state: SelectionState, ids: readonly string[], selected: boolean): SelectionState => {
  // Selecting in "all" mode means un-excluding; deselecting means excluding.
  const add = selected !== state.all
  const next = new Set(state.ids)
  for (const id of ids) {
    if (add) next.add(id)
    else next.delete(id)
  }
  return { ...state, ids: next }
}

/** Select every id from the anchor to `id` (inclusive) in `orderedIds`. Without a usable anchor, it selects just `id`. */
export const selectRange = (state: SelectionState, orderedIds: readonly string[], id: string): SelectionState => {
  const to = orderedIds.indexOf(id)
  const from = state.anchor === null ? -1 : orderedIds.indexOf(state.anchor)
  if (to === -1 || from === -1) return { ...setMany(state, [id], true), anchor: id }
  // The anchor stays put so a follow-up shift-click re-ranges from the same item.
  return setMany(state, orderedIds.slice(Math.min(from, to), Math.max(from, to) + 1), true)
}

export const replaceSelection = (ids: readonly string[]): SelectionState => ({
  ids: new Set(ids),
  anchor: null,
  all: false,
})

/** Select everything the current filter matches (the ids are resolved by the server, not listed here). */
export const selectAllMatching = (): SelectionState => ({ ids: new Set(), anchor: null, all: true })

/** The item filter a bulk request resolves server-side; mirrors the listing's filter params. */
export interface MatchFilter {
  split?: 'train' | 'validation' | 'test'
  classId?: string
  search?: string
  labeled?: boolean
}

/** The `itemIds` / `filter` half of a bulk request body. */
export type SelectionRequest = { itemIds: string[] } | { filter: MatchFilter; excludeItemIds: string[] }

/** What the server accepts as exclusions alongside a filter. */
export const MAX_EXCLUSIONS = 1000

/** The most items one filter-based bulk request may act on (the server's `MAX_FILTER_BULK`). */
export const MAX_FILTER_BULK = 5000

export const toRequest = (state: SelectionState, filter: MatchFilter): SelectionRequest =>
  state.all ? { filter, excludeItemIds: [...state.ids] } : { itemIds: [...state.ids] }
