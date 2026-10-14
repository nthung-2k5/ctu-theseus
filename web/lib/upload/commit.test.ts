import { expect, test } from 'bun:test'
import type { StagedItem } from '../uploadQueue'
import { newClassNamesOf, planFileCommit, planItemCommit, toInlinePayload } from './commit'
import { seededRandom } from './splitPlan'
import type { StagedFile } from './types'

const settings = { ratios: { train: 80, validation: 10, test: 10 }, stratify: true }
const options = { settings, taskUsesClasses: true, labelledWithoutClass: false, random: seededRandom(5) }

const file = (id: string, extra: Partial<StagedFile> = {}): StagedFile => ({
  id,
  batchId: 1,
  file: new File([id], `${id}.png`),
  name: `${id}.png`,
  size: 100,
  sourceDirs: [],
  classKey: 'c:cat',
  split: 'auto',
  extraDepth: 0,
  issues: [],
  ...extra,
})

test('files with an error stay staged; skipped ones leave without being sent', () => {
  const plan = planFileCommit(
    [
      file('ok'),
      file('bad', { issues: [{ severity: 'error', message: 'Empty file' }] }),
      file('dup', { skip: 'Already in this dataset' }),
      file('warn', { issues: [{ severity: 'warning', message: 'odd' }] }),
    ],
    options,
  )
  expect(plan.units.map((u) => u.id)).toEqual(['ok', 'warn'])
  expect(plan.consumedIds).toEqual(['ok', 'dup', 'warn'])
  expect(plan.alreadyThere).toBe(1)
})

test('an explicit split is kept and only auto files are drawn, per class', () => {
  const files = [
    ...Array.from({ length: 10 }, (_, i) => file(`cat${i}`, { classKey: 'c:cat' })),
    ...Array.from({ length: 10 }, (_, i) => file(`dog${i}`, { classKey: 'c:dog' })),
    file('pinned', { classKey: 'c:cat', split: 'test' }),
  ]
  const plan = planFileCommit(files, options)
  const bySplit = (prefix: string) => {
    const counts = { train: 0, validation: 0, test: 0 }
    for (const u of plan.units) if (u.id.startsWith(prefix)) counts[u.split]++
    return counts
  }
  expect(bySplit('cat')).toEqual({ train: 8, validation: 1, test: 1 })
  expect(bySplit('dog')).toEqual({ train: 8, validation: 1, test: 1 })
  expect(plan.units.find((u) => u.id === 'pinned')?.split).toBe('test')
  expect(plan.drawn?.counts).toEqual({ train: 16, validation: 2, test: 2 })
})

test('no draw happens when every file already has a split', () => {
  const plan = planFileCommit([file('a', { split: 'train' }), file('b', { split: 'validation' })], options)
  expect(plan.drawn).toBeNull()
  expect(plan.units.map((u) => u.split)).toEqual(['train', 'validation'])
})

test('a task without classes draws without grouping and sends no class', () => {
  const plan = planFileCommit(
    Array.from({ length: 10 }, (_, i) => file(`f${i}`, { classKey: null })),
    {
      ...options,
      taskUsesClasses: false,
      labelledWithoutClass: true,
    },
  )
  expect(plan.drawn?.counts).toEqual({ train: 8, validation: 1, test: 1 })
  expect(plan.units.every((u) => u.payload.classKey === null && u.labelled)).toBe(true)
})

test('a file is labelled when it has a class; weight is its size', () => {
  const plan = planFileCommit([file('a'), file('b', { classKey: null, size: 7 })], options)
  expect(plan.units.map((u) => [u.labelled, u.weight])).toEqual([
    [true, 100],
    [false, 7],
  ])
})

const item = (id: string, extra: Partial<StagedItem> = {}): StagedItem => ({
  id,
  kind: 'text',
  name: id,
  split: 'auto',
  classId: null,
  text: `text ${id}`,
  ...extra,
})

test('inline items are shaped for the items endpoint', () => {
  expect(toInlinePayload(item('a', { classId: 'k' }))).toEqual({
    textFeatures: { rawText: 'text a' },
    annotations: [{ annotationType: 'classification', classId: 'k' }],
  })
  expect(
    toInlinePayload(item('b', { kind: 'csv', text: undefined, featuresJson: { x: 1 }, targetValue: 2.5 })),
  ).toEqual({
    tabularFeatures: { featuresJson: { x: 1 } },
    annotations: [{ annotationType: 'classification', labelStructured: { value: 2.5 } }],
  })
  expect(toInlinePayload(item('c')).annotations).toBeUndefined()
})

test('inline items draw their split by class and are labelled by class or target', () => {
  const items = [
    ...Array.from({ length: 10 }, (_, i) => item(`p${i}`, { classId: 'pos' })),
    item('loose'),
    item('target', { kind: 'csv', targetValue: 3 }),
  ]
  const { units, drawn } = planItemCommit(items, options)
  const pos = units.filter((u) => u.id.startsWith('p'))
  expect(pos.filter((u) => u.split === 'validation')).toHaveLength(1)
  expect(drawn?.splits.size).toBe(12)
  expect(units.find((u) => u.id === 'loose')?.labelled).toBe(false)
  expect(units.find((u) => u.id === 'target')?.labelled).toBe(true)
  expect(pos.every((u) => u.labelled)).toBe(true)
})

test('only classes that a file is still sent to are created', () => {
  const pending = new Map([
    ['n:bird', { key: 'n:bird', name: 'Bird' }],
    ['n:fish', { key: 'n:fish', name: 'Fish' }],
  ])
  const plan = planFileCommit(
    [
      file('a', { classKey: 'n:bird' }),
      file('b', { classKey: 'n:bird' }),
      file('c', { classKey: 'n:fish', issues: [{ severity: 'error', message: 'bad' }] }),
      file('d', { classKey: 'c:cat' }),
    ],
    options,
  )
  expect(newClassNamesOf(plan.units, pending)).toEqual(['Bird'])
})
