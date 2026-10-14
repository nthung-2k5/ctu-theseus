/**
 * Upload jobs: a committed batch that keeps going after the add-data drawer is closed.
 *
 * Pressing Upload moves the units out of the staging area into a job. A job owns what is still to be sent, sends it
 * through a runner (files and inline items each have one), and keeps whatever failed or was never sent so it can be
 * retried. The store is module-level and framework-free; `useUploadJobs.ts` is the React side and `UploadTray` the UI.
 * It is in memory only: a page reload ends a job, and the tray warns before that happens.
 */

import type { SplitType } from '../../store/types'

/** One thing to send: a file or an inline item whose split has already been decided. */
export interface JobUnit<P> {
  id: string
  /** What the failure list calls it. */
  name: string
  split: SplitType
  /** False for an item that will land without a class or label (the result card offers to label those). */
  labelled: boolean
  /** Bytes for a file, 1 for an inline item: progress and the time left are weighted by it. */
  weight: number
  payload: P
}

export interface JobSettlement<P> {
  ok?: JobUnit<P>[]
  failed?: { unit: JobUnit<P>; reason: string }[]
  /** Of `ok`, how many the server already had (it stored nothing new). */
  duplicates?: number
  createdClasses?: number
}

export interface JobRunContext<P> {
  signal: AbortSignal
  /** Reports what a request settled. Call it as results arrive, not once at the end. */
  settle: (settlement: JobSettlement<P>) => void
  /** A short note for the tray ("Creating classes"), or `null` to clear it. */
  phase: (label: string | null) => void
}

export interface JobSpec<P> {
  projectId: string
  label: string
  units: JobUnit<P>[]
  /** Left out before the job started because the dataset already had their content. */
  alreadyThere?: number
  /** Sends `units`, reporting through `ctx.settle`. Also used for a retry, with only the units still owed. */
  run: (units: JobUnit<P>[], ctx: JobRunContext<P>) => Promise<void>
  /** A run ended (finished, cancelled or failed): refresh what the upload changed. */
  onFinished?: (job: UploadJobView) => void
}

export type JobStatus = 'running' | 'done' | 'cancelled'

export interface JobFailure {
  id: string
  name: string
  reason: string
}

export interface UploadJobView {
  id: string
  projectId: string
  label: string
  status: JobStatus
  phase: string | null
  /** Units in the job, including any that are retried. */
  total: number
  uploaded: number
  /** Content the dataset already had: skipped before the job, or reported by the server. */
  duplicates: number
  failed: number
  createdClasses: number
  /** Of the units uploaded so far. */
  bySplit: Record<SplitType, number>
  unlabelled: number
  failures: JobFailure[]
  /** What a retry would send: the failures plus anything a cancel left unsent. */
  retryable: number
  /** 0..1 of the current run, by weight. */
  fraction: number
  startedAt: number
  finishedAt: number | null
}

interface JobState<P> {
  spec: JobSpec<P>
  view: UploadJobView
  /** Everything not yet uploaded. */
  pending: Map<string, JobUnit<P>>
  failed: Map<string, string>
  controller: AbortController | null
  run: { weightTotal: number; weightSettled: number; startedAt: number }
}

// The state is typed per payload, but the store lists them together.
// biome-ignore lint/suspicious/noExplicitAny: heterogeneous payloads share one list; each job only touches its own
const states = new Map<string, JobState<any>>()
const listeners = new Set<() => void>()
let snapshot: UploadJobView[] = []

function publish() {
  snapshot = [...states.values()].map((s) => s.view)
  for (const listener of listeners) listener()
}

export const getJobs = (): UploadJobView[] => snapshot

export function subscribeJobs(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function refresh<P>(state: JobState<P>) {
  const { view, run } = state
  state.view = {
    ...view,
    failed: state.failed.size,
    retryable: state.pending.size,
    failures: [...state.failed].map(([id, reason]) => ({
      id,
      reason,
      name: state.pending.get(id)?.name ?? id,
    })),
    fraction: run.weightTotal > 0 ? Math.min(1, run.weightSettled / run.weightTotal) : 1,
  }
}

const describe = (error: unknown): string => (error instanceof Error ? error.message : 'Upload failed')

async function execute<P>(state: JobState<P>) {
  const units = [...state.pending.values()]
  state.failed.clear()
  const controller = new AbortController()
  state.controller = controller
  const weightTotal = units.reduce((n, u) => n + u.weight, 0)
  state.run = { weightTotal, weightSettled: 0, startedAt: Date.now() }
  state.view = { ...state.view, status: 'running', phase: null, finishedAt: null, startedAt: Date.now() }
  refresh(state)
  publish()

  const ctx: JobRunContext<P> = {
    signal: controller.signal,
    phase: (label) => {
      state.view = { ...state.view, phase: label }
      publish()
    },
    settle: ({ ok = [], failed = [], duplicates = 0, createdClasses = 0 }) => {
      const bySplit = { ...state.view.bySplit }
      let unlabelled = state.view.unlabelled
      for (const unit of ok) {
        if (!state.pending.delete(unit.id)) continue
        bySplit[unit.split]++
        if (!unit.labelled) unlabelled++
        state.run.weightSettled += unit.weight
      }
      for (const { unit, reason } of failed) {
        if (!state.pending.has(unit.id) || state.failed.has(unit.id)) continue
        state.failed.set(unit.id, reason)
        state.run.weightSettled += unit.weight
      }
      state.view = {
        ...state.view,
        bySplit,
        unlabelled,
        uploaded: state.view.uploaded + ok.length,
        duplicates: state.view.duplicates + duplicates,
        createdClasses: state.view.createdClasses + createdClasses,
      }
      refresh(state)
      publish()
    },
  }

  try {
    await state.spec.run(units, ctx)
  } catch (error) {
    // A runner that throws (not an abort) fails everything it had not already reported.
    if (!controller.signal.aborted) {
      const reason = describe(error)
      for (const unit of state.pending.values()) if (!state.failed.has(unit.id)) state.failed.set(unit.id, reason)
    }
  }

  const cancelled = controller.signal.aborted
  if (!cancelled) {
    // Nothing is silently dropped: a unit the runner never reported is a failure the user can retry.
    for (const unit of state.pending.values()) {
      if (!state.failed.has(unit.id)) state.failed.set(unit.id, 'The upload ended before this was sent')
    }
  }
  state.controller = null
  state.view = { ...state.view, status: cancelled ? 'cancelled' : 'done', phase: null, finishedAt: Date.now() }
  refresh(state)
  publish()
  try {
    state.spec.onFinished?.(state.view)
  } catch {
    // refreshing the page's data must not undo the job's own outcome
  }
}

export function startUploadJob<P>(spec: JobSpec<P>): string {
  const id = crypto.randomUUID()
  const state: JobState<P> = {
    // The units move into `pending`, which shrinks as they are uploaded; keeping them on the spec as well would hold
    // every file (an unpacked archive member lives in memory) until the result is dismissed.
    spec: { ...spec, units: [] },
    pending: new Map(spec.units.map((unit) => [unit.id, unit])),
    failed: new Map(),
    controller: null,
    run: { weightTotal: 0, weightSettled: 0, startedAt: Date.now() },
    view: {
      id,
      projectId: spec.projectId,
      label: spec.label,
      status: 'running',
      phase: null,
      total: spec.units.length,
      uploaded: 0,
      duplicates: spec.alreadyThere ?? 0,
      failed: 0,
      createdClasses: 0,
      bySplit: { train: 0, validation: 0, test: 0 },
      unlabelled: 0,
      failures: [],
      retryable: spec.units.length,
      fraction: 0,
      startedAt: Date.now(),
      finishedAt: null,
    },
  }
  states.set(id, state)
  void execute(state)
  return id
}

export function cancelJob(id: string): void {
  states.get(id)?.controller?.abort()
}

/** Sends again what is still owed: failures and anything a cancel left unsent. */
export function retryJob(id: string): void {
  const state = states.get(id)
  if (!state || state.controller || state.pending.size === 0) return
  void execute(state)
}

/** Forgets a finished job. Whatever it still owed is gone with it. */
export function dismissJob(id: string): void {
  const state = states.get(id)
  if (!state || state.controller) return
  states.delete(id)
  publish()
}

export const hasRunningJob = (): boolean => snapshot.some((job) => job.status === 'running')

/** Seconds left in the current run, from the pace so far; `null` until there is enough to go on. */
export function secondsLeft(job: UploadJobView, now: number): number | null {
  if (job.status !== 'running' || job.fraction <= 0 || job.fraction >= 1) return null
  const elapsed = (now - job.startedAt) / 1000
  if (elapsed < 2) return null
  return Math.ceil((elapsed / job.fraction) * (1 - job.fraction))
}
