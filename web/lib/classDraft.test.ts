import { expect, test } from 'bun:test'
import {
  attention,
  type DraftClass,
  filterRows,
  fromSaved,
  isDirty,
  looksLikeList,
  nameError,
  nextColor,
  parseBulk,
  removedFrom,
  resolveMerges,
  restoreRow,
  sortRows,
  summarize,
  toSaveBody,
  usedSplits,
} from './classDraft'

const row = (name: string, extra: Partial<DraftClass> = {}): DraftClass => ({
  key: extra.classId ?? `new-${name}`,
  name,
  description: '',
  color: '#00afef',
  ...extra,
})
const saved = (id: string, name: string) => row(name, { classId: id, key: id })

test('names are trimmed, required, unique ignoring case and whitespace, and a row may keep its own', () => {
  const draft = [saved('a', 'Cat'), saved('b', 'Dog')]
  expect(nameError('  ', draft)).toBe('Name is required')
  expect(nameError('x'.repeat(101), draft)).toBe('At most 100 characters')
  expect(nameError('  cat ', draft)).toBe('Already exists')
  expect(nameError('c  a t', draft)).toBeNull()
  expect(nameError('cat', draft, 'a')).toBeNull() // renaming a row to itself is fine
})

test('bulk parsing splits on newlines, commas and semicolons and reports what it skipped', () => {
  const draft = [saved('a', 'Cat')]
  expect(parseBulk('cow, pig\nhen;  CAT \n\n cow ', draft)).toEqual({
    names: ['cow', 'pig', 'hen'],
    skipped: ['CAT', 'cow'],
  })
  expect(parseBulk('y'.repeat(101), []).skipped).toHaveLength(1)
})

test('only a list-shaped paste is offered the bulk dialog', () => {
  expect(looksLikeList('cat')).toBe(false)
  expect(looksLikeList('cat\ndog')).toBe(true)
  expect(looksLikeList('cat, dog')).toBe(true)
  expect(looksLikeList('cat\n')).toBe(false) // a trailing newline alone is still one name
})

test('a new class takes the first palette colour nobody uses', () => {
  expect(nextColor([])).toBe('#00afef')
  expect(nextColor([row('a', { color: '#00AFEF' })])).toBe('#f59e0b') // compared case-insensitively
})

test('dirty means the rows differ from the saved list in content or length', () => {
  const baseline = [saved('a', 'Cat'), saved('b', 'Dog')]
  expect(
    isDirty(
      baseline,
      baseline.map((r) => ({ ...r })),
    ),
  ).toBe(false)
  expect(isDirty(baseline, [baseline[0]])).toBe(true)
  expect(isDirty(baseline, [baseline[0], { ...baseline[1], name: 'Dogs' }])).toBe(true)
  expect(isDirty(baseline, [baseline[0], { ...baseline[1], color: '#00AFEF' }])).toBe(false) // hex case isn't a change
})

test('removed classes are the saved ones missing from the draft, and restoring keeps the saved order', () => {
  const baseline = [saved('a', 'A'), saved('b', 'B'), saved('c', 'C')]
  const added = row('Fresh')
  const rows = [baseline[0], baseline[2], added]
  expect(removedFrom(baseline, rows).map((r) => r.key)).toEqual(['b'])
  expect(restoreRow(baseline, rows, baseline[1]).map((r) => r.name)).toEqual(['A', 'B', 'C', 'Fresh'])
})

test('a merge into a kept class is sent as it is', () => {
  const r = resolveMerges(['x'], { x: 'k' }, new Set(['k']))
  expect(r).toEqual({ merges: [{ fromClassId: 'x', intoClassId: 'k' }], unresolved: [] })
})

test('a merge whose target was removed afterwards follows the chain to a kept class', () => {
  const r = resolveMerges(['a', 'b'], { a: 'b', b: 'c' }, new Set(['c']))
  expect(r.merges).toEqual([
    { fromClassId: 'a', intoClassId: 'c' },
    { fromClassId: 'b', intoClassId: 'c' },
  ])
  expect(r.unresolved).toEqual([])
})

test('a chain that ends in a removed class with no target is unresolved, not silently dropped', () => {
  const r = resolveMerges(['a', 'b'], { a: 'b' }, new Set(['k']))
  expect(r.merges).toEqual([])
  expect(r.unresolved).toEqual(['a'])
})

test('a removal with no target is left alone, and a cycle cannot loop forever', () => {
  expect(resolveMerges(['a'], {}, new Set(['k']))).toEqual({ merges: [], unresolved: [] })
  expect(resolveMerges(['a', 'b'], { a: 'b', b: 'a' }, new Set(['k'])).unresolved).toEqual(['a', 'b'])
})

test('the summary counts additions, removals, renames and merges', () => {
  const baseline = [saved('a', 'A'), saved('b', 'B'), saved('c', 'C')]
  const rows = [{ ...baseline[0], name: 'A2' }, baseline[1], row('New')]
  expect(summarize(baseline, rows, 1)).toEqual({ added: 1, removed: 1, renamed: 1, merged: 1 })
})

test('the save body trims names, blanks empty descriptions, and only sends ids for saved classes', () => {
  const body = toSaveBody(
    [saved('a', ' Cat  '), row('New', { description: ' x ' })],
    [{ fromClassId: 'z', intoClassId: 'a' }],
  )
  expect(body.classes[0]).toEqual({ classId: 'a', name: 'Cat', description: null, uiColorHex: '#00afef' })
  expect(body.classes[1]).toEqual({ name: 'New', description: 'x', uiColorHex: '#00afef' })
  expect(body.merges).toEqual([{ fromClassId: 'z', intoClassId: 'a' }])
})

test('fromSaved falls back to a neutral colour and an empty description', () => {
  expect(fromSaved({ classId: 'a', name: 'A', description: null, uiColorHex: null })).toMatchObject({
    description: '',
    color: '#64748b',
  })
})

const counts = (total: number, train = total, validation = 0, test = 0) => ({ total, train, validation, test })

test('attention: empty beats few beats a missing split, and a split nobody uses is not a problem', () => {
  const inUse = usedSplits([counts(5, 3, 1, 1)])
  expect(attention(counts(0), 3, inUse)).toEqual({ kind: 'empty' })
  expect(attention(counts(2, 2), 3, inUse)).toEqual({ kind: 'few' }) // not also "missing validation"
  expect(attention(counts(4, 4), 3, inUse)).toEqual({ kind: 'missing-split', splits: ['validation', 'test'] })
  expect(attention(counts(5, 3, 1, 1), 3, inUse)).toBeNull()
  expect(attention(undefined, 3, inUse)).toBeNull() // counts not loaded yet
})

test('before any validation or test split exists, no class is flagged for lacking one', () => {
  const inUse = usedSplits([counts(5), counts(4)]) // everything is still in train
  expect([...inUse]).toEqual(['train'])
  expect(attention(counts(5), 3, inUse)).toBeNull()
})

test('filtering matches name or description, ignoring case and spacing', () => {
  const rows = [row('Cat'), row('Dog', { description: 'Barks a lot' })]
  expect(filterRows(rows, ' cat ').map((r) => r.name)).toEqual(['Cat'])
  expect(filterRows(rows, 'BARKS').map((r) => r.name)).toEqual(['Dog'])
  expect(filterRows(rows, '')).toBe(rows)
})

test('sorting is a view: it returns a copy and leaves creation order alone', () => {
  const rows = [row('b'), row('A'), row('c')]
  const totals: Record<string, number> = { b: 5, A: 0, c: 9 }
  const total = (r: DraftClass) => totals[r.name]
  expect(sortRows(rows, 'created', total, () => false)).toBe(rows)
  expect(sortRows(rows, 'name', total, () => false).map((r) => r.name)).toEqual(['A', 'b', 'c'])
  expect(sortRows(rows, 'count', total, () => false).map((r) => r.name)).toEqual(['c', 'b', 'A'])
  expect(rows.map((r) => r.name)).toEqual(['b', 'A', 'c'])
})

test('sorting by attention puts the problem classes first, the emptiest leading', () => {
  const rows = [row('ok1'), row('few'), row('ok2'), row('empty')]
  const totals: Record<string, number> = { ok1: 9, few: 2, ok2: 9, empty: 0 }
  const sorted = sortRows(
    rows,
    'attention',
    (r) => totals[r.name],
    (r) => totals[r.name] < 3,
  )
  expect(sorted.map((r) => r.name)).toEqual(['empty', 'few', 'ok1', 'ok2'])
})
