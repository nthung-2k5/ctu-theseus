import { expect, test } from 'bun:test'
import type { Annotation } from '@public/store/types'
import {
  backgroundMarker,
  boxesOf,
  boxesSummary,
  boxFromDrag,
  classColor,
  hitTest,
  isBackgroundMarker,
  MIN_BOX_SIZE,
  nudgeBox,
  pointOnImage,
  readBox,
} from './boxes'

const ann = (partial: Partial<Annotation>) =>
  ({ id: 'a', annotationType: 'bounding_box', ...partial }) as unknown as Annotation
const BOX = { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }

test('readBox accepts a well-formed box and nothing else', () => {
  expect(readBox(BOX)).toEqual(BOX)
  expect(readBox({ background: true })).toBeNull()
  expect(readBox({ x: 0.1, y: 0.2, w: 0, h: 0.4 })).toBeNull() // no width
  expect(readBox({ x: 0.1, y: 0.2, w: 0.3 })).toBeNull() // no height
  expect(readBox({ x: '0.1', y: 0.2, w: 0.3, h: 0.4 })).toBeNull()
  expect(readBox({ x: Number.NaN, y: 0.2, w: 0.3, h: 0.4 })).toBeNull()
  expect(readBox(null)).toBeNull()
  expect(readBox([0.1, 0.2, 0.3, 0.4])).toBeNull()
})

test('the marker is recognised only when it is exactly {background: true}', () => {
  expect(isBackgroundMarker({ background: true })).toBe(true)
  expect(isBackgroundMarker({ background: 1 })).toBe(false)
  expect(isBackgroundMarker(BOX)).toBe(false)
  expect(isBackgroundMarker(undefined)).toBe(false)
})

test('boxesOf keeps only bounding-box annotations that hold a box, in order, with their class', () => {
  const list = [
    ann({ id: '1', classId: 'cat', labelStructured: BOX }),
    ann({ id: 'x', annotationType: 'classification', labelStructured: BOX }), // another type: ignored
    ann({ id: 'm', classId: null, labelStructured: { background: true } }), // the marker is not a box
    ann({ id: 'bad', classId: 'cat', labelStructured: { x: 2 } }), // malformed: skipped, not a crash
    ann({ id: '2', classId: 'dog', labelStructured: { ...BOX, x: 0.5 } }),
  ]
  expect(boxesOf(list).map((b) => [b.id, b.classId])).toEqual([
    ['1', 'cat'],
    ['2', 'dog'],
  ])
  expect(boxesOf(undefined)).toEqual([])
})

test('the summary tells unlabelled, empty and boxed images apart', () => {
  expect(boxesSummary(undefined)).toBeNull() // nobody has labelled it
  expect(boxesSummary([])).toBeNull()
  expect(boxesSummary([ann({ labelStructured: { background: true } })])).toBe('No objects')
  expect(boxesSummary([ann({ id: '1', classId: 'c', labelStructured: BOX })])).toBe('1 box')
  expect(
    boxesSummary([
      ann({ id: '1', classId: 'c', labelStructured: BOX }),
      ann({ id: '2', classId: 'c', labelStructured: BOX }),
    ]),
  ).toBe('2 boxes')
  expect(backgroundMarker([ann({ id: 'm', labelStructured: { background: true } })])?.id).toBe('m')
  expect(backgroundMarker([ann({ labelStructured: BOX })])).toBeUndefined()
})

test('pointOnImage turns pointer coordinates into fractions and clamps them to the image', () => {
  const rect = { left: 100, top: 50, width: 200, height: 100 }
  expect(pointOnImage(200, 100, rect)).toEqual({ x: 0.5, y: 0.5 })
  expect(pointOnImage(50, 0, rect)).toEqual({ x: 0, y: 0 }) // dragged past the top-left corner
  expect(pointOnImage(999, 999, rect)).toEqual({ x: 1, y: 1 })
  expect(pointOnImage(10, 10, { left: 0, top: 0, width: 0, height: 0 })).toEqual({ x: 0, y: 0 }) // not laid out yet
})

test('a drag makes the same box whichever corner it starts from', () => {
  const a = { x: 0.2, y: 0.3 }
  const b = { x: 0.6, y: 0.8 }
  const expected = { x: 0.2, y: 0.3, w: 0.4, h: 0.5 }
  const close = (box: ReturnType<typeof boxFromDrag>) => {
    expect(box).not.toBeNull()
    for (const key of ['x', 'y', 'w', 'h'] as const) expect(box?.[key]).toBeCloseTo(expected[key], 10)
  }
  close(boxFromDrag(a, b))
  close(boxFromDrag(b, a))
  close(boxFromDrag({ x: 0.6, y: 0.3 }, { x: 0.2, y: 0.8 }))
})

test('a drag that is too small in either direction is a click, not a box', () => {
  expect(boxFromDrag({ x: 0.5, y: 0.5 }, { x: 0.5, y: 0.5 })).toBeNull()
  expect(boxFromDrag({ x: 0.2, y: 0.2 }, { x: 0.2 + MIN_BOX_SIZE / 2, y: 0.9 })).toBeNull() // a line, not an area
  expect(boxFromDrag({ x: 0.2, y: 0.2 }, { x: 0.9, y: 0.2 + MIN_BOX_SIZE / 2 })).toBeNull()
  expect(boxFromDrag({ x: 0.2, y: 0.2 }, { x: 0.2 + MIN_BOX_SIZE, y: 0.2 + MIN_BOX_SIZE })).not.toBeNull()
})

test('a box dragged to the image edges is still a valid box for the server', () => {
  const box = boxFromDrag({ x: 0, y: 0 }, { x: 1, y: 1 })
  expect(box).toEqual({ x: 0, y: 0, w: 1, h: 1 })
})

test('nudging moves a box but never out of the image', () => {
  expect(nudgeBox(BOX, 0.05, -0.1)).toEqual({ x: 0.15000000000000002, y: 0.1, w: 0.3, h: 0.4 })
  expect(nudgeBox(BOX, -1, -1)).toEqual({ x: 0, y: 0, w: 0.3, h: 0.4 })
  const far = nudgeBox(BOX, 5, 5)
  expect(far.x + far.w).toBeCloseTo(1, 10)
  expect(far.y + far.h).toBeCloseTo(1, 10)
})

test('hit testing picks the smallest box under the point, so nested boxes stay reachable', () => {
  const big = { id: 'big', classId: 'a', box: { x: 0, y: 0, w: 1, h: 1 } }
  const small = { id: 'small', classId: 'b', box: { x: 0.4, y: 0.4, w: 0.2, h: 0.2 } }
  expect(hitTest([big, small], { x: 0.5, y: 0.5 })?.id).toBe('small')
  expect(hitTest([small, big], { x: 0.5, y: 0.5 })?.id).toBe('small') // order does not matter
  expect(hitTest([big, small], { x: 0.1, y: 0.1 })?.id).toBe('big')
  expect(hitTest([small], { x: 0.1, y: 0.1 })).toBeNull()
  expect(hitTest([], { x: 0.5, y: 0.5 })).toBeNull()
})

test('a class is drawn in its own colour, else in a colour that only depends on its id', () => {
  const own = new Map([['cat', '#123456']])
  expect(classColor('cat', own)).toBe('#123456')
  expect(classColor('dog', own)).toBe(classColor('dog')) // stable: the same id gives the same colour every time
  expect(classColor('dog', own)).toMatch(/^#[0-9a-f]{6}$/i)
  expect(classColor(null)).toBe(classColor(undefined)) // no class: one neutral colour
  expect(classColor('dog')).not.toBe(classColor(null))
})
