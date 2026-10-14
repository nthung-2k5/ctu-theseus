import { expect, test } from 'bun:test'
import {
  buildIndex,
  type ClassRef,
  existingKey,
  fileIssues,
  initialState,
  isUploadable,
  type StageOptions,
  stageBatch,
  treeReducer,
} from './fileTree'
import { flattenTree } from './rows'
import type { RawEntry, StagedFile } from './types'

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
  // `cat/x.jpg` has no split folder: its split is left to be drawn at upload, and it is flagged as one folder short.
  expect(byName['x.jpg'].split).toBe('auto')
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

test('files without a split folder wait in auto, and only show as a bucket while something is in it', () => {
  const state = stage(raw('cat/a.jpg', 'cat/b.jpg'))
  expect([...state.files.values()].map((f) => f.split)).toEqual(['auto', 'auto'])
  const index = buildIndex(state, classes, true)
  expect(index.totals.bySplit).toEqual({ train: 0, validation: 0, test: 0, auto: 2 })
  const splits = (i: typeof index) =>
    flattenTree(i.folders, {}, true).flatMap((r) => (r.type === 'split' ? [r.split] : []))
  // Only the populated class is open (the empty one stays collapsed), and its `auto` bucket is listed last.
  expect(splits(index)).toEqual(['train', 'validation', 'test', 'auto'])

  const placed = treeReducer(state, { type: 'move', ids: [...state.files.keys()], split: 'validation' })
  expect(splits(buildIndex(placed, classes, true))).toEqual(['train', 'validation', 'test'])
})

test('hashes are recorded, and skip reasons set and cleared as a whole', () => {
  let state = stage(raw('cat/train/a.jpg', 'cat/train/b.jpg'))
  const [a, b] = [...state.files.keys()]
  state = treeReducer(state, { type: 'hashed', hashes: { [a]: 'h', [b]: 'h', nope: 'x' } })
  expect(state.files.get(a)?.hash).toBe('h')
  expect(state.files.has('nope')).toBe(false)

  state = treeReducer(state, { type: 'skip', reasons: { [b]: 'Same content as a.jpg' } })
  expect(state.files.get(b)?.skip).toBe('Same content as a.jpg')
  expect(state.files.get(a)?.skip).toBeUndefined()
  const totals = buildIndex(state, classes, true).totals
  expect(totals).toMatchObject({ files: 2, uploadable: 1, skipped: 1, withErrors: 0 })
  expect(isUploadable(state.files.get(b) as StagedFile)).toBe(false)
  expect(fileIssues(state.files.get(b) as StagedFile, true).map((i) => i.message)).toContain(
    'Same content as a.jpg, so it is not sent',
  )

  // Its twin was removed: the next plan has no reason for it, so the skip goes.
  state = treeReducer(state, { type: 'skip', reasons: {} })
  expect(state.files.get(b)?.skip).toBeUndefined()
  expect(buildIndex(state, classes, true).totals.skipped).toBe(0)
})

test('a file that is both broken and a duplicate is counted as broken, not skipped', () => {
  const staged = stageBatch(raw('cat/train/a.jpg'), [[{ severity: 'error', message: 'bad' }]], 1, options)
  let state = treeReducer(initialState(), { type: 'add', staged, junk: 0, skipped: [] })
  const [id] = [...state.files.keys()]
  state = treeReducer(state, { type: 'skip', reasons: { [id]: 'Already in this dataset' } })
  expect(buildIndex(state, classes, true).totals).toMatchObject({ withErrors: 1, skipped: 0, uploadable: 0 })
})
