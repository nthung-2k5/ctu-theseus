import { expect, test } from 'bun:test'
import { candidateLayouts, detectLayout, placeEntry, splitFromName } from './layout'
import { isJunkPath, normalizeClassName, normalizePath, sniffArchive, sniffMime } from './paths'

const all = candidateLayouts(true, {})

test('split aliases ignore case', () => {
  expect(splitFromName('Train')).toBe('train')
  expect(splitFromName('VAL')).toBe('validation')
  expect(splitFromName('dev')).toBe('validation')
  expect(splitFromName('Testing')).toBe('test')
  expect(splitFromName('cat')).toBeNull()
})

test('detects class/split', () => {
  const dirs = [
    ['cat', 'train'],
    ['cat', 'val'],
    ['dog', 'train'],
    ['dog', 'test'],
  ]
  expect(detectLayout(dirs, all)).toEqual({ kind: 'class/split', strip: 0 })
})

test('detects split/class (ImageFolder)', () => {
  const dirs = [
    ['train', 'cat'],
    ['train', 'dog'],
    ['val', 'cat'],
  ]
  expect(detectLayout(dirs, all).kind).toBe('split/class')
})

test('detects class-only, split-only and flat', () => {
  expect(detectLayout([['cat'], ['dog']], all).kind).toBe('class')
  expect(detectLayout([['train'], ['test']], all).kind).toBe('split')
  expect(detectLayout([[], []], all).kind).toBe('flat')
  expect(detectLayout([['a', 'b', 'c']], candidateLayouts(false, {})).kind).toBe('flat')
})

test('strips a wrapper folder only when that fits better', () => {
  const wrapped = [
    ['pets', 'cat', 'train'],
    ['pets', 'dog', 'test'],
  ]
  expect(detectLayout(wrapped, all)).toEqual({ kind: 'class/split', strip: 1 })
  // A lone class must stay a class instead of being treated as a wrapper.
  expect(
    detectLayout(
      [
        ['cat', 'train'],
        ['cat', 'test'],
      ],
      all,
    ),
  ).toEqual({ kind: 'class/split', strip: 0 })
})

test('drop context removes the levels it already fixes', () => {
  expect(candidateLayouts(true, { classKey: 'c:1' })).toEqual(['split', 'flat'])
  expect(candidateLayouts(true, { split: 'train' })).toEqual(['class', 'flat'])
  expect(candidateLayouts(true, { classKey: null, split: 'test' })).toEqual(['flat'])
  expect(candidateLayouts(false, {})).toEqual(['split', 'flat'])
})

test('places entries and counts leftover depth', () => {
  expect(placeEntry(['cat', 'train'], 'class/split', 0)).toEqual({ className: 'cat', split: 'train', extraDepth: 0 })
  expect(placeEntry(['pets', 'cat', 'val', 'x', 'y'], 'class/split', 1)).toEqual({
    className: 'cat',
    split: 'validation',
    extraDepth: 2,
  })
  // A file whose folders don't fit the layout keeps what matched.
  expect(placeEntry(['cat', 'misc'], 'class/split', 0)).toEqual({ className: 'cat', split: null, extraDepth: 1 })
  expect(placeEntry([], 'class/split', 0)).toEqual({ className: null, split: null, extraDepth: 0 })
})

test('normalizePath rejects escapes and cleans separators', () => {
  expect(normalizePath('cat\\train//./a.jpg')).toEqual({ ok: true, segments: ['cat', 'train', 'a.jpg'] })
  expect(normalizePath('../evil.jpg').ok).toBe(false)
  expect(normalizePath('a/../../b').ok).toBe(false)
  expect(normalizePath('/').ok).toBe(false)
})

test('junk and class-name normalization', () => {
  expect(isJunkPath(['__MACOSX', 'cat', 'a.jpg'])).toBe(true)
  expect(isJunkPath(['cat', '.DS_Store'])).toBe(true)
  expect(isJunkPath(['cat', 'Thumbs.db'])).toBe(true)
  expect(isJunkPath(['cat', 'a.jpg'])).toBe(false)
  expect(normalizeClassName('Golden_Retriever')).toBe('golden retriever')
  expect(normalizeClassName(' golden-retriever ')).toBe('golden retriever')
})

test('magic-byte sniffing', () => {
  expect(sniffMime(Uint8Array.from([0xff, 0xd8, 0xff, 0xe0]))).toBe('image/jpeg')
  expect(sniffMime(Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))).toBe('image/png')
  expect(sniffMime(new TextEncoder().encode('RIFF0000WAVEfmt '))).toBe('audio/wav')
  expect(sniffMime(new TextEncoder().encode('hello world!'))).toBeNull()
  expect(sniffArchive(Uint8Array.from([0x50, 0x4b, 0x03, 0x04]))).toBe('zip')
  expect(sniffArchive(Uint8Array.from([0x1f, 0x8b, 0x08]))).toBe('gzip')
  expect(sniffArchive(new TextEncoder().encode('not an archive'))).toBeNull()
})
