/**
 * The class lanes of the spot-check view: one lane per class (plus one for files with no class), each holding the
 * staged files that will land in it. Pure, so how lanes are ordered and sampled can be tested without a browser.
 */

import { type FolderNode, STAGED_SPLITS } from './fileTree'
import type { StagedFile, StagedSplit } from './types'

export interface Lane {
  key: string
  name: string
  /** A folder that will be created as a class when the batch is uploaded. */
  pending: boolean
  /** The lane for files with no class (or the single lane of a task without classes). */
  isNone: boolean
  /** Files in staging order: train, validation, test, then auto-split. */
  files: StagedFile[]
  bySplit: Record<StagedSplit, number>
}

export function buildLanes(folders: FolderNode[]): Lane[] {
  const lanes = folders.map(
    (folder): Lane => ({
      key: folder.key,
      name: folder.name,
      pending: folder.pending,
      isNone: folder.isNone,
      files: STAGED_SPLITS.flatMap((split) => folder.files[split]),
      bySplit: {
        train: folder.files.train.length,
        validation: folder.files.validation.length,
        test: folder.files.test.length,
        auto: folder.files.auto.length,
      },
    }),
  )
  // What needs a decision comes first: files with no class, then the lanes that have files; the empty classes (still
  // useful as places to drag a misfiled picture) go last. The order within each group is the folders' own.
  const rank = (lane: Lane) => (lane.isNone && lane.files.length > 0 ? 0 : lane.files.length > 0 ? 1 : 2)
  return lanes
    .map((lane, i) => ({ lane, i }))
    .sort((a, b) => rank(a.lane) - rank(b.lane) || a.i - b.i)
    .map(({ lane }) => lane)
}

/**
 * `count` items spread evenly through `list`, in order. A lane is checked by eye, and the first dozen files of a
 * folder tend to come from one sub-folder (or one camera); a spread catches the odd one out anywhere in it.
 */
export function sampleEvenly<T>(list: readonly T[], count: number): T[] {
  if (count <= 0) return []
  if (list.length <= count) return [...list]
  return Array.from({ length: count }, (_, i) => list[Math.floor((i * list.length) / count)])
}

/** The tiles a lane shows: an even sample while collapsed, then the first `limit` files once the user asks for more. */
export function laneTiles<T>(list: readonly T[], limit: number | null, sample: number): T[] {
  return limit === null ? sampleEvenly(list, sample) : list.slice(0, limit)
}

/**
 * Lets at most `limit` async jobs run at once, the rest waiting their turn in order. Decoding a photo into a small
 * tile is heavy, and a screenful of lanes asks for dozens of them in the same frame.
 */
export function createLimiter(limit: number) {
  let running = 0
  const waiting: (() => void)[] = []
  // A slot is claimed here, before the waiter resumes, so a burst of finishing jobs cannot release more waiters than
  // there are free slots.
  const release = () => {
    while (running < limit && waiting.length > 0) {
      running++
      waiting.shift()?.()
    }
  }
  return async function run<T>(job: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (running < limit && waiting.length === 0) running++
    else await new Promise<void>((resolve) => waiting.push(resolve))
    try {
      // The tile scrolled away (or was removed) while it waited: do not start the work at all.
      signal?.throwIfAborted()
      return await job()
    } finally {
      running--
      release()
    }
  }
}
