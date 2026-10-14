import { expect, test } from 'bun:test'
import {
  cancelJob,
  dismissJob,
  getJobs,
  type JobRunContext,
  type JobUnit,
  retryJob,
  secondsLeft,
  startUploadJob,
  subscribeJobs,
  type UploadJobView,
} from './jobs'

const unit = (id: string, extra: Partial<JobUnit<null>> = {}): JobUnit<null> => ({
  id,
  name: `${id}.png`,
  split: 'train',
  labelled: true,
  weight: 1,
  payload: null,
  ...extra,
})

const jobOf = (id: string): UploadJobView => {
  const job = getJobs().find((j) => j.id === id)
  if (!job) throw new Error('no such job')
  return job
}

/** A promise a test settles by hand, to hold a runner mid-flight. */
function gate() {
  let open!: () => void
  const wait = new Promise<void>((resolve) => {
    open = resolve
  })
  return { wait, open }
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

test('a job reports progress as results arrive and ends done', async () => {
  const hold = gate()
  const seen: number[] = []
  const unsubscribe = subscribeJobs(() => seen.push(getJobs().length))
  let finished: UploadJobView | null = null
  const id = startUploadJob({
    projectId: 'p',
    label: '4 files',
    units: [
      unit('a', { split: 'train' }),
      unit('b', { split: 'validation' }),
      unit('c', { split: 'test', labelled: false }),
      unit('d'),
    ],
    alreadyThere: 2,
    run: async (units, ctx) => {
      ctx.settle({ ok: units.slice(0, 2), duplicates: 1 })
      await hold.wait
      ctx.settle({ ok: units.slice(2), createdClasses: 3 })
    },
    onFinished: (job) => {
      finished = job
    },
  })
  await tick()
  const mid = jobOf(id)
  expect(mid).toMatchObject({ status: 'running', uploaded: 2, duplicates: 3, retryable: 2, fraction: 0.5 })
  expect(mid.bySplit).toEqual({ train: 1, validation: 1, test: 0 })

  hold.open()
  await tick()
  const done = jobOf(id)
  expect(done).toMatchObject({ status: 'done', uploaded: 4, failed: 0, retryable: 0, createdClasses: 3, unlabelled: 1 })
  expect(done.bySplit).toEqual({ train: 2, validation: 1, test: 1 })
  expect(done.finishedAt).not.toBeNull()
  expect(finished as UploadJobView | null).toMatchObject({ id, status: 'done' })
  expect(seen.length).toBeGreaterThan(2)
  unsubscribe()
  dismissJob(id)
  expect(getJobs().some((j) => j.id === id)).toBe(false)
})

test('failures are kept with their reason and a retry sends only those', async () => {
  const calls: string[][] = []
  const id = startUploadJob({
    projectId: 'p',
    label: 'files',
    units: [unit('a'), unit('b'), unit('c')],
    run: async (units, ctx) => {
      calls.push(units.map((u) => u.id))
      if (calls.length === 1) {
        ctx.settle({ ok: [units[0]], failed: units.slice(1).map((u) => ({ unit: u, reason: `refused ${u.id}` })) })
      } else {
        ctx.settle({ ok: units })
      }
    },
  })
  await tick()
  const first = jobOf(id)
  expect(first).toMatchObject({ status: 'done', uploaded: 1, failed: 2, retryable: 2 })
  expect(first.failures).toEqual([
    { id: 'b', name: 'b.png', reason: 'refused b' },
    { id: 'c', name: 'c.png', reason: 'refused c' },
  ])

  retryJob(id)
  await tick()
  expect(calls).toEqual([
    ['a', 'b', 'c'],
    ['b', 'c'],
  ])
  expect(jobOf(id)).toMatchObject({ status: 'done', uploaded: 3, failed: 0, retryable: 0, failures: [] })
  dismissJob(id)
})

test('cancelling stops the run, keeps the unsent units retryable and does not call them failures', async () => {
  const hold = gate()
  let signal: AbortSignal | undefined
  const id = startUploadJob({
    projectId: 'p',
    label: 'files',
    units: [unit('a'), unit('b'), unit('c')],
    run: async (units, ctx: JobRunContext<null>) => {
      signal = ctx.signal
      ctx.settle({ ok: [units[0]] })
      await hold.wait
    },
  })
  await tick()
  cancelJob(id)
  hold.open()
  await tick()
  expect(signal?.aborted).toBe(true)
  expect(jobOf(id)).toMatchObject({ status: 'cancelled', uploaded: 1, failed: 0, retryable: 2 })

  retryJob(id)
  await tick()
  // The retry runs the remaining two (the runner above settles only the first of whatever it gets).
  expect(jobOf(id).uploaded).toBe(2)
  cancelJob(id)
  hold.open()
  await tick()
  dismissJob(id)
})

test('a runner that throws fails what it had not reported; one that forgets units is told so', async () => {
  const thrown = startUploadJob({
    projectId: 'p',
    label: 'x',
    units: [unit('a'), unit('b')],
    run: async (units, ctx) => {
      ctx.settle({ ok: [units[0]] })
      throw new Error('network down')
    },
  })
  const forgetful = startUploadJob({ projectId: 'p', label: 'y', units: [unit('z')], run: async () => {} })
  await tick()
  expect(jobOf(thrown)).toMatchObject({ status: 'done', uploaded: 1, failed: 1 })
  expect(jobOf(thrown).failures[0].reason).toBe('network down')
  expect(jobOf(forgetful).failures[0].reason).toContain('ended before')
  dismissJob(thrown)
  dismissJob(forgetful)
})

test('a running job cannot be dismissed or retried', async () => {
  const hold = gate()
  const id = startUploadJob({
    projectId: 'p',
    label: 'x',
    units: [unit('a')],
    run: async () => {
      await hold.wait
    },
  })
  dismissJob(id)
  retryJob(id)
  expect(getJobs().some((j) => j.id === id)).toBe(true)
  cancelJob(id)
  hold.open()
  await tick()
  dismissJob(id)
})

test('a bad onFinished cannot undo the outcome', async () => {
  const id = startUploadJob({
    projectId: 'p',
    label: 'x',
    units: [unit('a')],
    run: async (units, ctx) => ctx.settle({ ok: units }),
    onFinished: () => {
      throw new Error('boom')
    },
  })
  await tick()
  expect(jobOf(id).status).toBe('done')
  dismissJob(id)
})

test('time left follows the pace so far and is silent until there is one', () => {
  const base = { status: 'running', startedAt: 0, fraction: 0.25 } as UploadJobView
  expect(secondsLeft(base, 1000)).toBeNull() // under two seconds in
  expect(secondsLeft(base, 10_000)).toBe(30) // 10 s for a quarter: 30 s to go
  expect(secondsLeft({ ...base, fraction: 0 }, 10_000)).toBeNull()
  expect(secondsLeft({ ...base, status: 'done' }, 10_000)).toBeNull()
})
