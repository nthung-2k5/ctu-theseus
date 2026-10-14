import { expect, test } from 'bun:test'
import { buildIndex, initialState, stageBatch, treeReducer } from './fileTree'
import { buildLanes, createLimiter, laneTiles, sampleEvenly } from './lanes'
import type { RawEntry } from './types'

const classes = [
  { classId: '1', name: 'Cat' },
  { classId: '2', name: 'Dog' },
]
const raw = (...paths: string[]): RawEntry[] =>
  paths.map((p) => ({ path: p.split('/'), file: new File(['x'], p.split('/').pop() ?? '') }))

function lanesOf(...paths: string[]) {
  const entries = raw(...paths)
  const staged = stageBatch(
    entries,
    entries.map(() => []),
    1,
    { drop: {}, taskUsesClasses: true, classes },
  )
  const state = treeReducer(initialState(), { type: 'add', staged, junk: 0, skipped: [] })
  return buildLanes(buildIndex(state, classes, true).folders)
}

test('lanes with files come first, and files with no class lead them; empty classes go last', () => {
  const lanes = lanesOf('Dog/train/a.jpg', 'Dog/test/b.jpg', 'bird/train/c.jpg')
  expect(lanes.map((l) => l.name)).toEqual(['Dog', 'bird', 'Cat'])
  expect(lanes[1].pending).toBe(true)

  const loose = lanesOf('Dog/train/a.jpg')
  expect(loose.map((l) => l.name)).toEqual(['Dog', 'Cat'])
})

test('a lane lists its files split by split, auto last, with the counts', () => {
  const [lane] = lanesOf('Cat/test/t.jpg', 'Cat/train/a.jpg', 'Cat/b.jpg')
  expect(lane.files.map((f) => f.name)).toEqual(['a.jpg', 't.jpg', 'b.jpg'])
  expect(lane.bySplit).toEqual({ train: 1, validation: 0, test: 1, auto: 1 })
})

test('an even sample spreads across the list, in order, and never repeats', () => {
  const list = Array.from({ length: 100 }, (_, i) => i)
  const sample = sampleEvenly(list, 10)
  expect(sample).toEqual([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
  expect(sampleEvenly([1, 2, 3], 10)).toEqual([1, 2, 3])
  expect(sampleEvenly(list, 0)).toEqual([])
  expect(new Set(sampleEvenly(list, 99)).size).toBe(99)
})

test('a lane shows a sample until it is expanded, then the first files in order', () => {
  const list = Array.from({ length: 50 }, (_, i) => i)
  expect(laneTiles(list, null, 5)).toEqual([0, 10, 20, 30, 40])
  expect(laneTiles(list, 3, 5)).toEqual([0, 1, 2])
})

test('the limiter never runs more than its limit at once and keeps order', async () => {
  const limit = createLimiter(2)
  let active = 0
  let peak = 0
  const order: number[] = []
  const job = (n: number) =>
    limit(async () => {
      active++
      peak = Math.max(peak, active)
      order.push(n)
      await new Promise((r) => setTimeout(r, 5))
      active--
      return n
    })
  expect(await Promise.all([1, 2, 3, 4, 5, 6].map(job))).toEqual([1, 2, 3, 4, 5, 6])
  expect(peak).toBe(2)
  expect(order).toEqual([1, 2, 3, 4, 5, 6])
})

test('a job aborted while it waited never starts, and its slot passes on', async () => {
  const limit = createLimiter(1)
  const controller = new AbortController()
  let started = 0
  const first = limit(async () => {
    started++
    await new Promise((r) => setTimeout(r, 10))
  })
  const second = limit(async () => {
    started++
  }, controller.signal)
  const third = limit(async () => {
    started++
  })
  controller.abort()
  await first
  await expect(second).rejects.toThrow()
  await third
  expect(started).toBe(2)
})
