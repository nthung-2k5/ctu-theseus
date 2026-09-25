import { expect, test } from 'bun:test'
import {
  buildIndex,
  type ClassRef,
  existingKey,
  fileIssues,
  initialState,
  type StageOptions,
  stageBatch,
  treeReducer,
} from './fileTree'
import type { RawEntry } from './types'

const classes: ClassRef[] = [
  { classId: '1', name: 'Cat' },
  { classId: '2', name: 'Dog' },
]
const options: StageOptions = { drop: {}, taskUsesClasses: true, classes }

const raw = (...paths: string[]): RawEntry[] =>
  paths.map((p) => {
    const path = p.split('/')
    return { path, file: new File(['x'], path[path.length - 1]) }
  })

function stage(entries: RawEntry[], opts: StageOptions = options) {
  const staged = stageBatch(
    entries,
    entries.map(() => []),
    1,
    opts,
  )
  return treeReducer(initialState(), { type: 'add', staged, junk: 0, skipped: [] })
}

test('stages class/split folders, matching existing classes ignoring case', () => {
  const state = stage(raw('cat/train/a.jpg', 'dog/val/b.jpg', 'Bird/test/c.jpg', 'cat/x.jpg'))
  const byName = Object.fromEntries([...state.files.values()].map((f) => [f.name, f]))
  expect(byName['a.jpg'].classKey).toBe(existingKey('1'))
  expect(byName['a.jpg'].split).toBe('train')
  expect(byName['b.jpg'].classKey).toBe(existingKey('2'))
  expect(byName['b.jpg'].split).toBe('validation')
  expect(byName['c.jpg'].classKey).toBe('n:bird')
  expect([...state.pending.values()].map((p) => p.name)).toEqual(['Bird'])
  // `cat/x.jpg` has no split folder: it lands in train and is flagged as one folder short of the layout.
  expect(byName['x.jpg'].split).toBe('train')
})

test('dropping onto a folder fixes its levels', () => {
  const state = stage(raw('a.jpg', 'val/b.jpg'), {
    ...options,
    drop: { classKey: existingKey('2') },
  })
  const files = [...state.files.values()]
  expect(files.every((f) => f.classKey === existingKey('2'))).toBe(true)
  expect(files.find((f) => f.name === 'b.jpg')?.split).toBe('validation')

  const both = stage(raw('deep/a.jpg'), { ...options, drop: { classKey: existingKey('1'), split: 'test' } })
  const [file] = [...both.files.values()]
  expect([file.classKey, file.split, file.extraDepth]).toEqual([existingKey('1'), 'test', 1])
})

test('tasks without classes ignore class folders', () => {
  const state = stage(raw('train/a.jpg', 'test/b.jpg'), { drop: {}, taskUsesClasses: false, classes: [] })
  expect([...state.files.values()].map((f) => [f.classKey, f.split])).toEqual([
    [null, 'train'],
    [null, 'test'],
  ])
  const index = buildIndex(state, [], false)
  expect(index.folders).toHaveLength(1)
  expect(index.folders[0].isNone).toBe(true)
  expect(fileIssues([...state.files.values()][0], false)).toEqual([])
})

test('relayout re-reads the batch and garbage-collects pending folders', () => {
  const state = stage(raw('train/cat/a.jpg', 'train/bird/b.jpg'))
  // Detected as split/class; forcing class/split reads `train` as a class instead.
  expect(state.lastBatch?.kind).toBe('split/class')
  expect([...state.pending.keys()].sort()).toEqual(['n:bird'])
  const flipped = treeReducer(state, { type: 'relayout', kind: 'class/split', options })
  expect([...flipped.pending.keys()]).toEqual([])
  expect([...flipped.files.values()].every((f) => f.classKey === null)).toBe(true)
})

test('move, rename, map and remove', () => {
  let state = stage(raw('bird/train/a.jpg', 'bird/val/b.jpg'))
  const ids = [...state.files.keys()]

  state = treeReducer(state, { type: 'move', ids: [ids[0]], split: 'test' })
  expect(state.files.get(ids[0])?.split).toBe('test')

  // Renaming a new class onto an existing one merges into it and drops the pending folder.
  state = treeReducer(state, { type: 'renameFolder', key: 'n:bird', name: 'dog', classes })
  expect([...state.files.values()].every((f) => f.classKey === existingKey('2'))).toBe(true)
  expect(state.pending.size).toBe(0)

  state = treeReducer(state, { type: 'remove', ids: [ids[1]] })
  expect(state.files.size).toBe(1)
})

test('reconcile folds pending folders into classes that now exist', () => {
  const state = stage(raw('bird/train/a.jpg'))
  const after = treeReducer(state, { type: 'reconcile', classes: [...classes, { classId: '3', name: 'Bird' }] })
  expect([...after.files.values()][0].classKey).toBe(existingKey('3'))
  expect(after.pending.size).toBe(0)
})

test('index lists every existing class and counts problems', () => {
  const entries = raw('cat/train/a.jpg', 'bird/test/b.jpg')
  const issues = [[], [{ severity: 'error' as const, message: 'bad' }]]
  const staged = stageBatch(entries, issues, 1, options)
  const state = treeReducer(initialState(), { type: 'add', staged, junk: 0, skipped: [] })
  const index = buildIndex(state, classes, true)
  expect(index.folders.map((f) => f.name)).toEqual(['Cat', 'Dog', 'bird'])
  expect(index.folders[0].files.train).toHaveLength(1)
  // The bird file has an error, so its class isn't needed yet.
  expect(index.totals.newClassNames).toEqual([])
  expect(index.totals).toMatchObject({ files: 2, uploadable: 1, withErrors: 1, unclassified: 0 })
})
