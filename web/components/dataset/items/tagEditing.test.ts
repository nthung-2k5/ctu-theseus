import { expect, test } from 'bun:test'
import {
  fillUntagged,
  isComplete,
  isValidTag,
  MAX_TAG_LENGTH,
  sameTags,
  setRange,
  setTag,
  splitWords,
  strayInside,
  taggedCount,
  tagsForWords,
} from './tagEditing'

test('words are split exactly like Python str.split(): on Unicode whitespace and nothing else', () => {
  expect(splitWords('Mr. Anna lives in Hue , Vietnam.')).toEqual(['Mr.', 'Anna', 'lives', 'in', 'Hue', ',', 'Vietnam.'])
  expect(splitWords('  a\tb\nc   d　e f ')).toEqual(['a', 'b', 'c', 'd', 'e', 'f'])
  // Python counts these as whitespace and JavaScript's \s does not...
  expect(splitWords('a\u001cb\u0085c')).toEqual(['a', 'b', 'c'])
  // ...and the other way round: a BOM is not whitespace to Python, so it stays inside its word.
  expect(splitWords('a﻿b c')).toEqual(['a﻿b', 'c'])
  expect(splitWords('')).toEqual([])
  expect(splitWords(' \t\n ')).toEqual([])
  expect(splitWords('Đà Nẵng')).toEqual(['Đà', 'Nẵng'])
})

test('a tag is one non-empty word of at most 64 characters', () => {
  expect(isValidTag('B-PER')).toBe(true)
  expect(isValidTag('O')).toBe(true)
  expect(isValidTag('')).toBe(false)
  expect(isValidTag('B PER')).toBe(false)
  expect(isValidTag(' O')).toBe(false)
  expect(isValidTag('x'.repeat(MAX_TAG_LENGTH))).toBe(true)
  expect(isValidTag('x'.repeat(MAX_TAG_LENGTH + 1))).toBe(false)
  // Characters beyond the BMP are one character each, as in Python.
  expect(isValidTag('😀'.repeat(MAX_TAG_LENGTH))).toBe(true)
})

test('stored tags are shown only when there is exactly one per word', () => {
  expect(tagsForWords(['O', 'B-PER'], 2)).toEqual(['O', 'B-PER'])
  expect(tagsForWords(['O', 'B-PER'], 3)).toEqual(['', '', ''])
  expect(tagsForWords(null, 2)).toEqual(['', ''])
  expect(tagsForWords(['O', 1], 2)).toEqual(['', ''])
  expect(tagsForWords({ background: true }, 0)).toEqual([])
  // A copy, never the stored array itself.
  const stored = ['O']
  expect(tagsForWords(stored, 1)).not.toBe(stored)
})

test('setting tags returns new arrays and covers a range in either direction', () => {
  const tags = ['', '', '', '']
  expect(setTag(tags, 1, 'O')).toEqual(['', 'O', '', ''])
  expect(tags).toEqual(['', '', '', ''])
  expect(setRange(tags, 1, 3, 'I-LOC')).toEqual(['', 'I-LOC', 'I-LOC', 'I-LOC'])
  expect(setRange(tags, 3, 1, 'I-LOC')).toEqual(['', 'I-LOC', 'I-LOC', 'I-LOC'])
})

test('filling leaves tagged words alone and completeness needs every word', () => {
  expect(fillUntagged(['B-PER', '', 'I-PER', ''], 'O')).toEqual(['B-PER', 'O', 'I-PER', 'O'])
  expect(taggedCount(['B-PER', '', 'O'])).toBe(2)
  expect(isComplete(['B-PER', 'O'])).toBe(true)
  expect(isComplete(['B-PER', ''])).toBe(false)
  // A text with no words has nothing to save.
  expect(isComplete([])).toBe(false)
  expect(sameTags(['O'], ['O'])).toBe(true)
  expect(sameTags(['O'], ['O', 'O'])).toBe(false)
  expect(sameTags(['O', 'B'], ['O', 'I'])).toBe(false)
})

test('an I- tag that continues nothing is pointed out', () => {
  expect(strayInside(['O', 'B-PER', 'I-PER', 'O', 'I-LOC'])).toEqual([4])
  expect(strayInside(['I-PER', 'O'])).toEqual([0])
  expect(strayInside(['B-PER', 'I-LOC'])).toEqual([1])
  expect(strayInside(['B-LOC', 'I-LOC', 'I-LOC'])).toEqual([])
  expect(strayInside(['O', 'PERSON', 'LOC'])).toEqual([])
})
