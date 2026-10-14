import { expect, test } from 'bun:test'
import { canHash, hashFiles, IN_DATASET_REASON, planSkips, SAME_AS_PREFIX, sha256Hex } from './preflight'

test('sha256 matches the well-known digest of "abc"', async () => {
  expect(canHash()).toBe(true)
  expect(await sha256Hex(new Blob(['abc']))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
})

test('hashFiles reports every digest, progress by bytes, and survives an unreadable file', async () => {
  const broken = new File(['xx'], 'broken.png')
  broken.arrayBuffer = () => Promise.reject(new Error('gone'))
  const files = [
    { id: 'a', file: new File(['abc'], 'a.png') },
    { id: 'b', file: new File(['abc'], 'b.png') },
    { id: 'x', file: broken },
    { id: 'c', file: new File(['other'], 'c.png') },
  ]
  const seen: string[] = []
  const progress: number[] = []
  const hashes = await hashFiles(files, { onHashed: (id) => seen.push(id), onProgress: (f) => progress.push(f) })
  expect([...hashes.keys()].sort()).toEqual(['a', 'b', 'c'])
  expect(hashes.get('a')).toBe(hashes.get('b'))
  expect(hashes.get('a')).not.toBe(hashes.get('c'))
  expect(seen.sort()).toEqual(['a', 'b', 'c'])
  expect(progress.at(-1)).toBe(1)
  expect(progress).toEqual([...progress].sort((x, y) => x - y))
})

test('hashFiles stops when aborted', async () => {
  const controller = new AbortController()
  controller.abort()
  await expect(
    hashFiles([{ id: 'a', file: new File(['abc'], 'a.png') }], { signal: controller.signal }),
  ).rejects.toThrow()
})

test('planSkips keeps the first copy and points later copies back at it', () => {
  const reasons = planSkips(
    [
      { id: '1', name: 'a.png', hash: 'h1' },
      { id: '2', name: 'b.png', hash: 'h2' },
      { id: '3', name: 'copy.png', hash: 'h1' },
      { id: '4', name: 'again.png', hash: 'h1' },
      { id: '5', name: 'unread.png' },
    ],
    new Set(),
  )
  expect(reasons).toEqual({ '3': `${SAME_AS_PREFIX}a.png`, '4': `${SAME_AS_PREFIX}a.png` })
})

test('a content the dataset already has is skipped wherever it appears, including its first copy', () => {
  const reasons = planSkips(
    [
      { id: '1', name: 'a.png', hash: 'h1' },
      { id: '2', name: 'b.png', hash: 'h2' },
      { id: '3', name: 'c.png', hash: 'h1' },
    ],
    new Set(['h1']),
  )
  expect(reasons).toEqual({ '1': IN_DATASET_REASON, '3': IN_DATASET_REASON })
})
