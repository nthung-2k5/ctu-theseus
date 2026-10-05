import { expect, test } from 'bun:test'
import { columnsFor, itemsInRow, rowCount } from './grid'

test('columns fit as many minimum-width items as the width allows, counting the gaps between them', () => {
  expect(columnsFor(1000, 180, 12)).toBe(5) // 5*180 + 4*12 = 948 <= 1000
  expect(columnsFor(948, 180, 12)).toBe(5)
  expect(columnsFor(947, 180, 12)).toBe(4)
})

test('there is always at least one column, including before the width is measured', () => {
  expect(columnsFor(0, 180, 12)).toBe(1)
  expect(columnsFor(100, 180, 12)).toBe(1)
})

test('rows round up, and the last row holds the remainder', () => {
  expect(rowCount(0, 4)).toBe(0)
  expect(rowCount(9, 4)).toBe(3)
  const items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  expect(itemsInRow(items, 4, 0)).toEqual([1, 2, 3, 4])
  expect(itemsInRow(items, 4, 2)).toEqual([9])
  expect(itemsInRow(items, 4, 3)).toEqual([])
})
