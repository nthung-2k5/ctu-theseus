import { expect, test } from 'bun:test'
import { type Box, nextInDirection } from './gridNav'

/** A rows x cols grid of 100x100 cards with 10px gaps, in reading order. */
const grid = (rows: number, cols: number): Box[] =>
  Array.from({ length: rows * cols }, (_, i) => ({
    left: (i % cols) * 110,
    top: Math.floor(i / cols) * 110,
    width: 100,
    height: 100,
  }))

test('left and right move along the row and stop at its ends', () => {
  const boxes = grid(2, 3)
  expect(nextInDirection(boxes, 1, 'right')).toBe(2)
  expect(nextInDirection(boxes, 1, 'left')).toBe(0)
  expect(nextInDirection(boxes, 2, 'right')).toBeNull() // no wrapping onto the next row
  expect(nextInDirection(boxes, 3, 'left')).toBeNull()
})

test('up and down keep the column and stop at the top and bottom', () => {
  const boxes = grid(3, 3)
  expect(nextInDirection(boxes, 4, 'down')).toBe(7)
  expect(nextInDirection(boxes, 4, 'up')).toBe(1)
  expect(nextInDirection(boxes, 1, 'up')).toBeNull()
  expect(nextInDirection(boxes, 7, 'down')).toBeNull()
})

test('moving down into a shorter last row lands on the closest card', () => {
  const boxes = grid(2, 3).slice(0, 4) // row 2 holds only one card, in column 0
  expect(nextInDirection(boxes, 2, 'down')).toBe(3)
})

test('it works whatever the column count, because it reads positions not indexes', () => {
  const narrow = grid(4, 2)
  expect(nextInDirection(narrow, 0, 'down')).toBe(2)
  expect(nextInDirection(narrow, 1, 'right')).toBeNull()
})

test('a card of a different height does not confuse the row it is in', () => {
  const boxes: Box[] = [
    { left: 0, top: 0, width: 100, height: 140 }, // taller card
    { left: 110, top: 0, width: 100, height: 100 },
    { left: 0, top: 150, width: 100, height: 100 },
    { left: 110, top: 150, width: 100, height: 100 },
  ]
  expect(nextInDirection(boxes, 0, 'right')).toBe(1)
  expect(nextInDirection(boxes, 1, 'down')).toBe(3)
  expect(nextInDirection(boxes, 0, 'down')).toBe(2)
})

test('a missing or lone card has nowhere to go', () => {
  expect(nextInDirection([], 0, 'right')).toBeNull()
  expect(nextInDirection(grid(1, 1), 0, 'down')).toBeNull()
})
