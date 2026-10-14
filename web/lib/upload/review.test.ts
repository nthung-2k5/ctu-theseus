import { expect, test } from 'bun:test'
import type { StagedItem } from '../uploadQueue'
import { countRows, fileToRow, filterRows, itemToRow, rowNote } from './review'
import type { StagedFile } from './types'

const file = (id: string, extra: Partial<StagedFile> = {}): StagedFile => ({
  id,
  batchId: 1,
  file: new File(['x'], `${id}.png`),
  name: `${id}.png`,
  size: 1,
  sourceDirs: [],
  classKey: 'c:1',
  split: 'train',
  extraDepth: 0,
  issues: [],
  ...extra,
})

test('a file row carries its problems, but the no-class and skip messages are left to their own columns', () => {
  const row = fileToRow(file('a', { classKey: null, skip: 'Already in this dataset', extraDepth: 2 }), true)
  expect(row.unclassified).toBe(true)
  expect(row.skip).toBe('Already in this dataset')
  expect(row.issues.map((i) => i.message)).toEqual(['2 deeper folder levels ignored'])
  expect(fileToRow(file('b', { classKey: null }), false).unclassified).toBe(false)
})

test('an inline row maps its class and flags a missing one only when the task wants classes', () => {
  const item: StagedItem = { id: 'i', kind: 'text', name: 'hello', split: 'auto', classId: null }
  expect(itemToRow(item, true)).toMatchObject({ classKey: null, unclassified: true, split: 'auto' })
  expect(itemToRow(item, false).unclassified).toBe(false)
  expect(itemToRow({ ...item, classId: 'k' }, true)).toMatchObject({ classKey: 'c:k', unclassified: false })
  // A regression row has a target, not a class, so it is never "missing" one.
  expect(itemToRow({ ...item, kind: 'csv', targetValue: 3 }, true).unclassified).toBe(false)
})

test('filters and their counts agree', () => {
  const rows = [
    fileToRow(file('ok'), true),
    fileToRow(file('bad', { issues: [{ severity: 'error', message: 'Empty file' }] }), true),
    fileToRow(file('warn', { issues: [{ severity: 'warning', message: 'odd' }], classKey: null }), true),
    fileToRow(file('dup', { skip: 'Already in this dataset', split: 'auto' }), true),
    fileToRow(
      file('brokendup', { skip: 'Already in this dataset', issues: [{ severity: 'error', message: 'x' }] }),
      true,
    ),
  ]
  const counts = countRows(rows)
  expect(counts).toEqual({ all: 5, errors: 2, warnings: 1, unclassified: 1, skipped: 1, auto: 1 })
  for (const filter of ['errors', 'warnings', 'unclassified', 'skipped', 'auto'] as const) {
    expect(filterRows(rows, filter)).toHaveLength(counts[filter])
  }
  expect(filterRows(rows, 'all')).toBe(rows)
  expect(filterRows(rows, 'skipped').map((r) => r.id)).toEqual(['dup'])
})

test('a readable image gets a thumbnail source, a broken or non-image file does not', () => {
  const png = (id: string, extra: Partial<StagedFile> = {}) =>
    file(id, { file: new File(['x'], `${id}.png`, { type: 'image/png' }), ...extra })
  expect(fileToRow(png('a'), true)).toMatchObject({ media: 'image' })
  expect(fileToRow(png('a'), true).thumb).toBeInstanceOf(File)
  expect(fileToRow(png('b', { issues: [{ severity: 'error', message: 'Empty file' }] }), true).thumb).toBeUndefined()
  const wav = fileToRow(file('c', { file: new File(['x'], 'c.wav', { type: 'audio/wav' }) }), true)
  expect(wav).toMatchObject({ media: 'audio' })
  expect(wav.thumb).toBeUndefined()
})

test('a row shows its worst problem as one note', () => {
  const issue = (severity: 'error' | 'warning', message: string) => ({ severity, message })
  expect(
    rowNote(fileToRow(file('a', { issues: [issue('warning', 'odd'), issue('error', 'Empty file')] }), true)),
  ).toEqual({
    text: 'Empty file',
    tone: 'error',
  })
  expect(rowNote(fileToRow(file('b', { issues: [issue('warning', 'odd')] }), true))).toEqual({
    text: 'odd',
    tone: 'warning',
  })
  expect(rowNote(fileToRow(file('c', { skip: 'Already in this dataset' }), true))).toEqual({
    text: 'Already in this dataset, so it is not sent',
    tone: 'muted',
  })
  expect(rowNote(fileToRow(file('d'), true))).toBeNull()
})
