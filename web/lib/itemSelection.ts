/**
 * Pure selection-state transitions for the item browser. Kept free of React so
 * they can be unit-tested; `hooks/useItemSelection.ts` wraps them in state.
 *
 * `anchor` is the last item toggled directly, the fixed end of a shift-click range.
 */
export interface SelectionState {
  ids: ReadonlySet<string>
  anchor: string | null
}

export const emptySelection: SelectionState = { ids: new Set(), anchor: null }

export const toggleItem = (state: SelectionState, id: string): SelectionState => {
  const ids = new Set(state.ids)
  if (ids.has(id)) ids.delete(id)
  else ids.add(id)
  return { ids, anchor: id }
}

/** Select every id from the anchor to `id` (inclusive) in `orderedIds`. Without a usable anchor, it selects just `id`. */
export const selectRange = (state: SelectionState, orderedIds: readonly string[], id: string): SelectionState => {
  const to = orderedIds.indexOf(id)
  const from = state.anchor === null ? -1 : orderedIds.indexOf(state.anchor)
  if (to === -1 || from === -1) {
    return { ids: new Set(state.ids).add(id), anchor: id }
  }
  const ids = new Set(state.ids)
  for (const rangeId of orderedIds.slice(Math.min(from, to), Math.max(from, to) + 1)) ids.add(rangeId)
  // The anchor stays put so a follow-up shift-click re-ranges from the same item.
  return { ids, anchor: state.anchor }
}

export const setMany = (state: SelectionState, ids: readonly string[], selected: boolean): SelectionState => {
  const next = new Set(state.ids)
  for (const id of ids) {
    if (selected) next.add(id)
    else next.delete(id)
  }
  return { ids: next, anchor: state.anchor }
}

export const replaceSelection = (ids: readonly string[]): SelectionState => ({
  ids: new Set(ids),
  anchor: null,
})
