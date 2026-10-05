import { expect, test } from 'bun:test'
import type { Annotation } from '@public/store/types'
import {
  CLASS_GROUND_TRUTH,
  findGroundTruth,
  type GroundTruthSpec,
  groundTruthKind,
  groundTruthLabel,
  targetValue,
} from './groundTruth'

const ann = (partial: Partial<Annotation>) => ({ id: 'a1', ...partial }) as unknown as Annotation

const REGRESSION: GroundTruthSpec = { type: 'classification', requiresLabelClasses: false }
const CAPTION: GroundTruthSpec = { type: 'text_sequence', requiresLabelClasses: false }
const TAGS: GroundTruthSpec = { type: 'token_tags', requiresLabelClasses: true }
const classes = new Map([['c1', 'cat']])

test('kind comes from the registry spec, so regression is a classification with no classes', () => {
  expect(groundTruthKind(CLASS_GROUND_TRUTH)).toBe('class')
  expect(groundTruthKind(REGRESSION)).toBe('number')
  expect(groundTruthKind(CAPTION)).toBe('text')
  expect(groundTruthKind(TAGS)).toBe('tags')
  expect(groundTruthKind({ type: 'bounding_box', requiresLabelClasses: true })).toBe('other')
})

test('findGroundTruth ignores annotations of another type', () => {
  const list = [ann({ id: 'x', annotationType: 'text_sequence' }), ann({ id: 'y', annotationType: 'classification' })]
  expect(findGroundTruth(list, CLASS_GROUND_TRUTH)?.id).toBe('y')
  expect(findGroundTruth(undefined, CLASS_GROUND_TRUTH)).toBeUndefined()
})

test('targetValue reads labelStructured.value and rejects anything that is not a finite number', () => {
  expect(targetValue(ann({ labelStructured: { value: 2.5 } }))).toBe(2.5)
  expect(targetValue(ann({ labelStructured: { value: 0 } }))).toBe(0)
  expect(targetValue(ann({ labelStructured: { value: '2.5' } }))).toBeNull()
  expect(targetValue(ann({ labelStructured: { value: Number.NaN } }))).toBeNull()
  expect(targetValue(ann({ labelStructured: null }))).toBeNull()
  expect(targetValue(undefined)).toBeNull()
})

test('label rendering per kind', () => {
  expect(
    groundTruthLabel([ann({ annotationType: 'classification', classId: 'c1' })], CLASS_GROUND_TRUTH, classes),
  ).toBe('cat')
  expect(
    groundTruthLabel([ann({ annotationType: 'classification', labelStructured: { value: 3 } })], REGRESSION, classes),
  ).toBe('3')
  expect(
    groundTruthLabel([ann({ annotationType: 'text_sequence', labelTextSequence: 'a dog' })], CAPTION, classes),
  ).toBe('a dog')
  expect(groundTruthLabel([ann({ annotationType: 'token_tags', labelStructured: ['O', 'B'] })], TAGS, classes)).toBe(
    '2 tags',
  )
  expect(groundTruthLabel([], CLASS_GROUND_TRUTH, classes)).toBeNull()
  // A class that is no longer in the lookup (deleted) renders as unlabeled rather than as a raw id.
  expect(
    groundTruthLabel([ann({ annotationType: 'classification', classId: 'gone' })], CLASS_GROUND_TRUTH, classes),
  ).toBeNull()
})
