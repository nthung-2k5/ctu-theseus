/**
 * Draws a split for every staged item whose split is still `auto`, when a batch is committed.
 *
 * It follows the server's `plan_split` (`services/datasets.py`): each group is shuffled and cut on its own, train and
 * validation round like JavaScript's `Math.round`, and test takes the remainder so every item lands in exactly one
 * split. With `stratify`, a group is a class, so a class cannot be shuffled entirely out of validation or test. It only
 * ever sees the items being added: what the draft already holds keeps its split.
 */

import type { SplitType } from '../../store/types'

export interface Ratios {
  train: number
  validation: number
  test: number
}

export const DEFAULT_RATIOS: Ratios = { train: 80, validation: 10, test: 10 }

/** Below this a class cannot appear in all three splits; the same threshold as the server's `MIN_ITEMS_PER_CLASS`. */
export const MIN_ITEMS_PER_GROUP = 3

export const ratioTotal = (ratios: Ratios): number => ratios.train + ratios.validation + ratios.test

/** A ratio set `planSplits` can use: no negatives or non-numbers, and something to divide by. */
export const validRatios = (ratios: Ratios): boolean =>
  [ratios.train, ratios.validation, ratios.test].every((n) => Number.isFinite(n) && n >= 0) && ratioTotal(ratios) > 0

export interface SplitCandidate {
  id: string
  /** The class the item is stratified by (`null` = no class). Ignored unless `stratify` is set. */
  group: string | null
}

export interface SplitPlan {
  splits: Map<string, SplitType>
  counts: Record<SplitType, number>
  /** Groups with fewer than `MIN_ITEMS_PER_GROUP` items, which cannot reach every split. Only when stratified. */
  smallGroups: string[]
}

/** A small seeded generator (mulberry32), so a plan can be reproduced in tests. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function shuffle<T>(values: T[], random: () => number): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[values[i], values[j]] = [values[j], values[i]]
  }
}

const UNASSIGNED = '\u0000unassigned'

export function planSplits(
  items: SplitCandidate[],
  ratios: Ratios,
  stratify: boolean,
  random: () => number = Math.random,
): SplitPlan {
  if (!validRatios(ratios)) throw new Error('Ratios must be non-negative and sum to more than zero')
  const total = ratioTotal(ratios)

  const groups = new Map<string, string[]>()
  for (const { id, group } of items) {
    const key = stratify ? (group ?? UNASSIGNED) : ''
    const ids = groups.get(key)
    if (ids) ids.push(id)
    else groups.set(key, [id])
  }

  const plan: SplitPlan = {
    splits: new Map(),
    counts: { train: 0, validation: 0, test: 0 },
    smallGroups: [],
  }
  for (const [key, ids] of groups) {
    shuffle(ids, random)
    const trainEnd = Math.round((ratios.train / total) * ids.length)
    const validationEnd = trainEnd + Math.round((ratios.validation / total) * ids.length)
    ids.forEach((id, i) => {
      const split: SplitType = i < trainEnd ? 'train' : i < validationEnd ? 'validation' : 'test'
      plan.splits.set(id, split)
      plan.counts[split]++
    })
    if (stratify && key !== UNASSIGNED && ids.length < MIN_ITEMS_PER_GROUP) plan.smallGroups.push(key)
  }
  return plan
}

/**
 * The concrete split of every unit: an explicit one stays, an `auto` one is drawn. `splitOf` and `groupOf` read the
 * caller's own shape, so files and inline items share this.
 */
export function resolveSplits<T extends { id: string }>(
  units: T[],
  options: {
    splitOf: (unit: T) => SplitType | 'auto'
    groupOf: (unit: T) => string | null
    ratios: Ratios
    stratify: boolean
    random?: () => number
  },
): { splits: Map<string, SplitType>; drawn: SplitPlan | null } {
  const splits = new Map<string, SplitType>()
  const toDraw: SplitCandidate[] = []
  for (const unit of units) {
    const split = options.splitOf(unit)
    if (split === 'auto') toDraw.push({ id: unit.id, group: options.groupOf(unit) })
    else splits.set(unit.id, split)
  }
  if (toDraw.length === 0) return { splits, drawn: null }
  const drawn = planSplits(toDraw, options.ratios, options.stratify, options.random)
  for (const [id, split] of drawn.splits) splits.set(id, split)
  return { splits, drawn }
}
