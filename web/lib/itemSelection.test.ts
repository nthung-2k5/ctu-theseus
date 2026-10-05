import { expect, test } from 'bun:test'
import { emptySelection, replaceSelection, selectRange, setMany, toggleItem } from './itemSelection'

const ids = ['a', 'b', 'c', 'd', 'e']
const sorted = (s: { ids: ReadonlySet<string> }) => [...s.ids].sort()

test('toggle adds, removes, and moves the anchor', () => {
  const on = toggleItem(emptySelection, 'b')
  expect(sorted(on)).toEqual(['b'])
  expect(on.anchor).toBe('b')
  const off = toggleItem(on, 'b')
  expect(sorted(off)).toEqual([])
})

test('range selects between the anchor and the target in either direction', () => {
  const forward = selectRange(toggleItem(emptySelection, 'b'), ids, 'd')
  expect(sorted(forward)).toEqual(['b', 'c', 'd'])
  const backward = selectRange(toggleItem(emptySelection, 'd'), ids, 'b')
  expect(sorted(backward)).toEqual(['b', 'c', 'd'])
})

test('range keeps the anchor and existing selection', () => {
  const start = toggleItem(toggleItem(emptySelection, 'a'), 'c') // anchor c, a also selected
  const next = selectRange(start, ids, 'e')
  expect(sorted(next)).toEqual(['a', 'c', 'd', 'e'])
  expect(next.anchor).toBe('c')
})

test('range with no anchor, or an anchor off this page, selects only the target', () => {
  expect(sorted(selectRange(emptySelection, ids, 'c'))).toEqual(['c'])
  const stale = toggleItem(emptySelection, 'zzz')
  expect(sorted(selectRange(stale, ids, 'c'))).toEqual(['c', 'zzz'])
})

test('setMany selects or deselects a group without inverting a mixed selection', () => {
  const mixed = toggleItem(emptySelection, 'a')
  expect(sorted(setMany(mixed, ids, true))).toEqual(ids)
  expect(sorted(setMany(setMany(mixed, ids, true), ['a', 'b'], false))).toEqual(['c', 'd', 'e'])
})

test('replaceSelection drops the anchor', () => {
  const next = replaceSelection(['x', 'y'])
  expect(sorted(next)).toEqual(['x', 'y'])
  expect(next.anchor).toBeNull()
})
