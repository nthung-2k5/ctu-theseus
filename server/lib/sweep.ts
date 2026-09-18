/**
 * Hyperparameter sweep search-space expansion.
 *
 * Deliberately not Ludwig's own `hyperopt` (Ray Tune-backed): that would
 * contend with training for the single GPU worker, its progress is opaque
 * to the run/metric/log event pipeline this app already has, and it reuses
 * none of the cancel/DLQ/reaper machinery `queueTraining` already gives a
 * manually-started run. Instead a sweep is just N ordinary training runs —
 * this module's only job is turning a search space into that list of N
 * `TrainerSelections`, which `routes/sweeps.ts` then dispatches one at a
 * time through the existing `queueTraining` (server/lib/microservice.ts).
 */

import type { TrainerSelections } from '@server/lib/ludwig'

/** One trainer knob → its candidate values for this sweep. Keys mirror TrainerSelections. */
export type SweepSearchSpace = Partial<Record<keyof TrainerSelections, (number | string)[]>>

const MAX_TRIALS_CAP = 50

export function validateSearchSpace(searchSpace: SweepSearchSpace, maxTrials: number): string | null {
  const keys = Object.keys(searchSpace)
  if (keys.length === 0) return 'Search space must specify at least one hyperparameter'
  for (const key of keys) {
    const values = searchSpace[key as keyof SweepSearchSpace]
    if (!values || values.length === 0) return `'${key}' must list at least one candidate value`
  }
  if (maxTrials < 1 || maxTrials > MAX_TRIALS_CAP) return `maxTrials must be between 1 and ${MAX_TRIALS_CAP}`
  return null
}

/** Every combination of the search space's candidate values (cartesian product), in a stable order. */
export function expandGrid(searchSpace: SweepSearchSpace): TrainerSelections[] {
  const entries = Object.entries(searchSpace) as [keyof TrainerSelections, (number | string)[]][]
  return entries.reduce<TrainerSelections[]>(
    (combos, [key, values]) => combos.flatMap((combo) => values.map((value) => ({ ...combo, [key]: value }))),
    [{}],
  )
}

/** `count` combinations, each knob sampled independently and uniformly from its candidates. */
export function sampleRandom(searchSpace: SweepSearchSpace, count: number): TrainerSelections[] {
  const entries = Object.entries(searchSpace) as [keyof TrainerSelections, (number | string)[]][]
  return Array.from({ length: count }, () => {
    // Built loosely-typed and cast once at the end, rather than per-key —
    // TrainerSelections' value type varies per key, but the candidate list
    // for each key was declared against that same key in SweepSearchSpace.
    const trial: Record<string, number | string> = {}
    for (const [key, values] of entries) {
      trial[key] = values[Math.floor(Math.random() * values.length)]
    }
    return trial as TrainerSelections
  })
}

/**
 * Expand a search space into the trials to dispatch. `grid` is capped at
 * `maxTrials` (truncated, not sampled down — a user who wants the full grid
 * sets maxTrials to the product size) so a large search space can't
 * accidentally dispatch hundreds of runs against the single GPU worker.
 */
export function expandSweep(
  searchSpace: SweepSearchSpace,
  strategy: 'grid' | 'random',
  maxTrials: number,
): TrainerSelections[] {
  if (strategy === 'grid') return expandGrid(searchSpace).slice(0, maxTrials)
  return sampleRandom(searchSpace, maxTrials)
}
