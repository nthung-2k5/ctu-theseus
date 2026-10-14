import { expect, test } from 'bun:test'
import { DEFAULT_RATIOS, planSplits, resolveSplits, seededRandom, validRatios } from './splitPlan'

const ids = (n: number, prefix = 'i') => Array.from({ length: n }, (_, i) => ({ id: `${prefix}${i}`, group: null }))

test('every item is assigned exactly once and test absorbs the rounding remainder', () => {
  const plan = planSplits(ids(10), { train: 70, validation: 15, test: 15 }, false, seededRandom(1))
  expect(plan.splits.size).toBe(10)
  // round(7) = 7, round(1.5) = 2 (halves round up, like the server), the remaining 1 is test.
  expect(plan.counts).toEqual({ train: 7, validation: 2, test: 1 })
  expect(plan.smallGroups).toEqual([])
})

test('the default ratios cut 100 items 80 / 10 / 10', () => {
  expect(planSplits(ids(100), DEFAULT_RATIOS, false, seededRandom(7)).counts).toEqual({
    train: 80,
    validation: 10,
    test: 10,
  })
})

test('the same seed reproduces the plan and another seed shuffles differently', () => {
  const a = planSplits(ids(50), DEFAULT_RATIOS, false, seededRandom(3))
  const b = planSplits(ids(50), DEFAULT_RATIOS, false, seededRandom(3))
  const c = planSplits(ids(50), DEFAULT_RATIOS, false, seededRandom(4))
  expect([...a.splits]).toEqual([...b.splits])
  expect([...a.splits]).not.toEqual([...c.splits])
})

test('stratified, each class is cut on its own so none is shuffled out of a split', () => {
  const items = [
    ...ids(20, 'cat').map((i) => ({ ...i, group: 'cat' })),
    ...ids(10, 'dog').map((i) => ({ ...i, group: 'dog' })),
  ]
  const plan = planSplits(items, DEFAULT_RATIOS, true, seededRandom(9))
  const of = (prefix: string, split: string) =>
    [...plan.splits].filter(([id, s]) => id.startsWith(prefix) && s === split).length
  expect([of('cat', 'train'), of('cat', 'validation'), of('cat', 'test')]).toEqual([16, 2, 2])
  expect([of('dog', 'train'), of('dog', 'validation'), of('dog', 'test')]).toEqual([8, 1, 1])
})

test('a class too small to reach every split is reported, an unclassed group is not', () => {
  const items = [
    { id: 'a', group: 'rare' },
    { id: 'b', group: 'rare' },
    { id: 'c', group: null },
    { id: 'd', group: 'common' },
    { id: 'e', group: 'common' },
    { id: 'f', group: 'common' },
  ]
  expect(planSplits(items, DEFAULT_RATIOS, true, seededRandom(1)).smallGroups).toEqual(['rare'])
  expect(planSplits(items, DEFAULT_RATIOS, false, seededRandom(1)).smallGroups).toEqual([])
})

test('ratios that cannot divide anything are refused', () => {
  expect(validRatios({ train: 0, validation: 0, test: 0 })).toBe(false)
  expect(validRatios({ train: -1, validation: 2, test: 0 })).toBe(false)
  expect(validRatios({ train: Number.NaN, validation: 1, test: 1 })).toBe(false)
  expect(validRatios({ train: 1, validation: 0, test: 0 })).toBe(true)
  expect(() => planSplits(ids(3), { train: 0, validation: 0, test: 0 }, false)).toThrow()
})

test('only auto items are drawn; an explicit split is left alone', () => {
  const units = [
    { id: 'a', split: 'test' as const },
    { id: 'b', split: 'auto' as const },
    { id: 'c', split: 'auto' as const },
    { id: 'd', split: 'validation' as const },
  ]
  const { splits, drawn } = resolveSplits(units, {
    splitOf: (u) => u.split,
    groupOf: () => null,
    ratios: { train: 1, validation: 0, test: 0 },
    stratify: false,
    random: seededRandom(1),
  })
  expect(splits.get('a')).toBe('test')
  expect(splits.get('d')).toBe('validation')
  expect(splits.get('b')).toBe('train')
  expect(splits.get('c')).toBe('train')
  expect(drawn?.counts.train).toBe(2)
})

test('nothing to draw means no plan (and no error on unusable ratios)', () => {
  const { drawn } = resolveSplits([{ id: 'a', split: 'train' as const }], {
    splitOf: (u) => u.split,
    groupOf: () => null,
    ratios: { train: 0, validation: 0, test: 0 },
    stratify: false,
  })
  expect(drawn).toBeNull()
})
