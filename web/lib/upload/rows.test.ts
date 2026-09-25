import { expect, test } from 'bun:test'
import { buildIndex, existingKey, initialState, type StageOptions, stageBatch, treeReducer } from './fileTree'
import { AUTO_OPEN_LIMIT, classRowKey, filesUnder, flattenTree, splitRowKey } from './rows'

const options: StageOptions = {
  drop: {},
  taskUsesClasses: true,
  classes: [
    { classId: '1', name: 'cat' },
    { classId: '2', name: 'dog' },
  ],
}

function indexOf(paths: string[]) {
  const raw = paths.map((p) => ({ path: p.split('/'), file: new File(['x'], p.split('/').pop() ?? '') }))
  const staged = stageBatch(
    raw,
    raw.map(() => []),
    1,
    options,
  )
  const state = treeReducer(initialState(), { type: 'add', staged, junk: 0, skipped: [] })
  return buildIndex(state, options.classes, true)
}

test('empty classes stay collapsed, populated ones open with all three splits', () => {
  const rows = flattenTree(indexOf(['cat/train/a.jpg', 'cat/test/b.jpg']).folders, {}, true)
  expect(
    rows.map((r) => `${r.type}:${r.type === 'file' ? r.file.name : r.type === 'split' ? r.split : r.node.name}`),
  ).toEqual(['class:cat', 'split:train', 'file:a.jpg', 'split:validation', 'split:test', 'file:b.jpg', 'class:dog'])
})

test('manual toggles win over the defaults', () => {
  const folders = indexOf(['cat/train/a.jpg']).folders
  const collapsed = flattenTree(folders, { [classRowKey(existingKey('1'))]: false }, true)
  expect(collapsed.map((r) => r.type)).toEqual(['class', 'class'])
  const empty = flattenTree(
    folders,
    { [splitRowKey(existingKey('2'), 'test')]: true, [classRowKey(existingKey('2'))]: true },
    true,
  )
  expect(empty.filter((r) => r.type === 'split' && r.node.name === 'dog')).toHaveLength(3)
})

test('big folders start collapsed and files are found under any row', () => {
  const paths = Array.from({ length: AUTO_OPEN_LIMIT + 1 }, (_, i) => `cat/train/${i}.jpg`)
  const folders = indexOf(paths).folders
  const rows = flattenTree(folders, {}, true)
  expect(rows.map((r) => r.type)).toEqual(['class', 'class'])
  const classRow = flattenTree(folders, { [classRowKey(existingKey('1'))]: true }, true)
  expect(classRow.filter((r) => r.type === 'file')).toHaveLength(0)
  expect(filesUnder(rows[0])).toHaveLength(AUTO_OPEN_LIMIT + 1)
})
