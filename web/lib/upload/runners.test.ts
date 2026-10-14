import { expect, test } from 'bun:test'
import { dismissJob, getJobs, type JobUnit, startUploadJob, type UploadJobView } from './jobs'
import { type InlineApi, type InlinePayload, ITEMS_PER_REQUEST, inlineRunner } from './uploadInline'
import {
  type FileApi,
  type FilePayload,
  type FileUnit,
  fileRunner,
  MAX_CHUNK_BYTES,
  MAX_CHUNK_FILES,
  toChunks,
} from './uploadStaged'

const fileUnit = (id: string, over: Partial<FileUnit> = {}, classKey: string | null = null): FileUnit => ({
  id,
  name: `${id}.png`,
  split: 'train',
  labelled: classKey !== null,
  weight: 10,
  payload: { file: new File([id], `${id}.png`), classKey },
  ...over,
})

const tick = () => new Promise((resolve) => setTimeout(resolve, 5))
const jobOf = (id: string): UploadJobView => {
  const job = getJobs().find((j) => j.id === id)
  if (!job) throw new Error('no such job')
  return job
}

interface Upload {
  split: string
  classId?: string
  names: string[]
}

function fakeFileApi(over: Partial<FileApi> = {}) {
  const uploads: Upload[] = []
  const created: string[] = []
  const api: FileApi = {
    listClasses: async () => [{ classId: 'c-cat', name: 'Cat' }],
    createClass: async (_p, name) => {
      created.push(name)
      return `new-${name}`
    },
    upload: async (_p, body) => {
      uploads.push({ split: body.split, classId: body.classId, names: body.files.map((f) => f.name) })
      return body.files.map(() => ({ status: 'fulfilled' }))
    },
    ...over,
  }
  return { api, uploads, created }
}

async function runFiles(units: FileUnit[], api: FileApi, pending = new Map<string, { key: string; name: string }>()) {
  const id = startUploadJob<FilePayload>({
    projectId: 'p',
    label: 'files',
    units,
    run: fileRunner({ api, projectId: 'p', pending }),
  })
  await tick()
  const job = jobOf(id)
  dismissJob(id)
  return job
}

test('files are grouped by split and class and cut into chunks of at most 50', () => {
  const units = [
    ...Array.from({ length: 120 }, (_, i) => fileUnit(`a${i}`, {}, 'c:1')),
    fileUnit('t', { split: 'test' }, 'c:1'),
    fileUnit('u', {}, null),
  ]
  const chunks = toChunks(units, (u) => (u.payload.classKey === 'c:1' ? '1' : null))
  expect(chunks.map((c) => [c.split, c.classId, c.units.length])).toEqual([
    ['train', '1', MAX_CHUNK_FILES],
    ['train', '1', MAX_CHUNK_FILES],
    ['train', '1', 20],
    ['test', '1', 1],
    ['train', null, 1],
  ])
})

test('a chunk also stops at the byte cap', () => {
  const half = MAX_CHUNK_BYTES / 2 + 1
  const chunks = toChunks([fileUnit('a', { weight: half }), fileUnit('b', { weight: half })], () => null)
  expect(chunks.map((c) => c.units.length)).toEqual([1, 1])
})

test('every file is uploaded once, to the class its folder named', async () => {
  const { api, uploads, created } = fakeFileApi()
  const job = await runFiles(
    [fileUnit('a', {}, 'c:c-cat'), fileUnit('b', { split: 'validation' }, 'c:c-cat'), fileUnit('c', {}, null)],
    api,
  )
  expect(created).toEqual([])
  expect(uploads.map((u) => [u.split, u.classId, u.names])).toEqual([
    ['train', 'c-cat', ['a.png']],
    ['validation', 'c-cat', ['b.png']],
    ['train', undefined, ['c.png']],
  ])
  expect(job).toMatchObject({ status: 'done', uploaded: 3, failed: 0 })
  expect(job.bySplit).toEqual({ train: 2, validation: 1, test: 0 })
  expect(job.unlabelled).toBe(1)
})

test('a new class is created once, reusing one that has appeared since, and counted', async () => {
  const { api, uploads, created } = fakeFileApi()
  const pending = new Map([
    ['n:bird', { key: 'n:bird', name: 'Bird' }],
    ['n:cat', { key: 'n:cat', name: 'cat' }], // the same class as the existing "Cat", ignoring case
  ])
  const job = await runFiles(
    [fileUnit('a', {}, 'n:bird'), fileUnit('b', {}, 'n:bird'), fileUnit('c', {}, 'n:cat')],
    api,
    pending,
  )
  expect(created).toEqual(['Bird'])
  expect(uploads.map((u) => [u.classId, u.names])).toEqual([
    ['new-Bird', ['a.png', 'b.png']],
    ['c-cat', ['c.png']],
  ])
  expect(job.createdClasses).toBe(1)
})

test('files whose class could not be created fail with the reason and are not sent unlabelled', async () => {
  const { api, uploads } = fakeFileApi({
    createClass: async () => {
      throw new Error('class name taken')
    },
  })
  const job = await runFiles(
    [fileUnit('a', {}, 'n:bird'), fileUnit('b', {}, 'c:c-cat')],
    api,
    new Map([['n:bird', { key: 'n:bird', name: 'Bird' }]]),
  )
  expect(uploads.map((u) => u.names)).toEqual([['b.png']])
  expect(job).toMatchObject({ uploaded: 1, failed: 1 })
  expect(job.failures).toEqual([{ id: 'a', name: 'a.png', reason: 'class name taken' }])
})

test('a file the server refuses fails alone, and duplicates are counted', async () => {
  const { api } = fakeFileApi({
    upload: async () => [
      { status: 'fulfilled', isDuplicate: true },
      { status: 'rejected', reason: 'File is larger than the 50 MB limit' },
      { status: 'fulfilled' },
    ],
  })
  const job = await runFiles([fileUnit('a'), fileUnit('b'), fileUnit('c')], api)
  expect(job).toMatchObject({ uploaded: 2, duplicates: 1, failed: 1 })
  expect(job.failures[0]).toMatchObject({ id: 'b', reason: 'File is larger than the 50 MB limit' })
})

test('a request that fails fails its whole chunk with the transport error', async () => {
  const { api } = fakeFileApi({
    upload: async () => {
      throw new Error('socket closed')
    },
  })
  const job = await runFiles([fileUnit('a'), fileUnit('b')], api)
  expect(job).toMatchObject({ uploaded: 0, failed: 2 })
  expect(job.failures.map((f) => f.reason)).toEqual(['socket closed', 'socket closed'])
})

test('inline items go out in chunks and only the refused ones fail', async () => {
  const sizes: number[] = []
  const api: InlineApi = {
    create: async (_p, items) => {
      sizes.push(items.length)
      return { failed: sizes.length === 1 ? [{ index: 1, message: 'bad row' }] : [] }
    },
  }
  const units: JobUnit<InlinePayload>[] = Array.from({ length: ITEMS_PER_REQUEST + 3 }, (_, i) => ({
    id: `r${i}`,
    name: `Row ${i}`,
    split: i === 0 ? 'test' : 'train',
    labelled: true,
    weight: 1,
    payload: { textFeatures: { rawText: `t${i}` } },
  }))
  const id = startUploadJob({ projectId: 'p', label: 'rows', units, run: inlineRunner({ api, projectId: 'p' }) })
  await tick()
  const job = jobOf(id)
  dismissJob(id)
  expect(sizes).toEqual([ITEMS_PER_REQUEST, 3])
  expect(job).toMatchObject({ status: 'done', uploaded: ITEMS_PER_REQUEST + 2, failed: 1 })
  expect(job.failures).toEqual([{ id: 'r1', name: 'Row 1', reason: 'bad row' }])
})

test('the split a unit carries is the split that is sent', async () => {
  const sent: string[] = []
  const api: InlineApi = {
    create: async (_p, items) => {
      sent.push(...items.map((i) => i.split))
      return { failed: [] }
    },
  }
  const unit = (id: string, split: 'train' | 'validation' | 'test'): JobUnit<InlinePayload> => ({
    id,
    name: id,
    split,
    labelled: false,
    weight: 1,
    payload: { tabularFeatures: { featuresJson: { a: 1 } } },
  })
  const id = startUploadJob({
    projectId: 'p',
    label: 'rows',
    units: [unit('a', 'validation'), unit('b', 'test')],
    run: inlineRunner({ api, projectId: 'p' }),
  })
  await tick()
  dismissJob(id)
  expect(sent).toEqual(['validation', 'test'])
})
