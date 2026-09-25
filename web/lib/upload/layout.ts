/**
 * Works out what a set of dropped folder paths *means*: which level is the class, which is the split.
 *
 * Every layout is a list of levels read from the top of the (wrapper-stripped) path. `class/split`
 * (`cat/train/a.jpg`) is the preferred one; the rest cover the other common ways datasets are laid out on
 * disk. Detection scores each candidate by how much of the files' folder paths it explains, and ties go to the
 * one with fewer levels.
 */

import type { SplitType } from '../../store/types'
import type { DropContext, LayoutKind, Level } from './types'

export const LAYOUT_LEVELS: Record<LayoutKind, Level[]> = {
  'class/split': ['class', 'split'],
  'split/class': ['split', 'class'],
  class: ['class'],
  split: ['split'],
  flat: [],
}

/** Candidate order = tie-break order. */
const CANDIDATES: LayoutKind[] = ['class/split', 'split/class', 'class', 'split', 'flat']

export const LAYOUT_LABELS: Record<LayoutKind, string> = {
  'class/split': 'class / split / file',
  'split/class': 'split / class / file',
  class: 'class / file',
  split: 'split / file',
  flat: 'files only',
}

const SPLIT_ALIASES: Record<string, SplitType> = {
  train: 'train',
  training: 'train',
  val: 'validation',
  valid: 'validation',
  validation: 'validation',
  dev: 'validation',
  test: 'test',
  testing: 'test',
}

/** The split a folder name stands for (`Val`, `valid`, `dev` → `validation`), or `null` if it isn't one. */
export function splitFromName(name: string): SplitType | null {
  return SPLIT_ALIASES[name.trim().toLowerCase()] ?? null
}

function fitsLevel(segment: string | undefined, level: Level): boolean {
  if (segment === undefined) return false
  return level === 'split' ? splitFromName(segment) !== null : splitFromName(segment) === null
}

/** How many consecutive levels, from the top of `dirs`, match `levels`. */
function consumed(dirs: string[], levels: Level[]): number {
  let n = 0
  for (const level of levels) {
    if (!fitsLevel(dirs[n], level)) break
    n++
  }
  return n
}

/** Layouts a drop can use: levels the drop target already fixed, or that the task doesn't have, are excluded. */
export function candidateLayouts(taskUsesClasses: boolean, ctx: DropContext): LayoutKind[] {
  const classFixed = ctx.classKey !== undefined || !taskUsesClasses
  const splitFixed = ctx.split !== undefined
  return CANDIDATES.filter((kind) => {
    const levels = LAYOUT_LEVELS[kind]
    return !(classFixed && levels.includes('class')) && !(splitFixed && levels.includes('split'))
  })
}

export interface LayoutDetection {
  kind: LayoutKind
  /** Leading folders shared by every file (`pets/…`) that carry no meaning and are dropped. */
  strip: number
}

const MAX_STRIP = 3

/**
 * Picks the layout that fits the most files. A wrapper folder (`pets/cat/train/a.jpg`) is stripped only when
 * that fits strictly better than leaving it, so a lone `cat/train/…` stays a class.
 */
export function detectLayout(dirsList: string[][], candidates: LayoutKind[]): LayoutDetection {
  if (dirsList.length === 0 || candidates.length === 0) return { kind: 'flat', strip: 0 }

  // Deduplicate directory paths: 50k files usually share a few hundred directories.
  const unique = new Map<string, { dirs: string[]; count: number }>()
  for (const dirs of dirsList) {
    const key = dirs.join('\u0000')
    const hit = unique.get(key)
    if (hit) hit.count++
    else unique.set(key, { dirs, count: 1 })
  }
  const groups = [...unique.values()]

  let common = Math.min(...groups.map((g) => g.dirs.length))
  const first = groups[0].dirs
  for (const g of groups) {
    let i = 0
    while (i < common && g.dirs[i] === first[i]) i++
    common = i
  }

  const EPSILON = 1e-9
  let best: { score: number; levels: number; detection: LayoutDetection } | null = null
  for (let strip = 0; strip <= Math.min(common, MAX_STRIP); strip++) {
    for (const kind of candidates) {
      const levels = LAYOUT_LEVELS[kind]
      // Coverage: the share of each file's folders the layout explains from the top. Partial matches count, so a
      // few loose files in `cat/` don't sink `class/split` for a dataset that is otherwise `cat/train/...`.
      let covered = 0
      for (const g of groups) {
        const rest = g.dirs.slice(strip)
        covered += g.count * (rest.length === 0 ? 1 : consumed(rest, levels) / rest.length)
      }
      const score = covered / dirsList.length
      // Higher coverage wins; at equal coverage a later strip never beats an earlier one, and fewer levels win
      // (`cat/a.jpg` is `class`, not a `class/split` with no split anywhere).
      const better =
        !best ||
        score > best.score + EPSILON ||
        (Math.abs(score - best.score) <= EPSILON && best.detection.strip === strip && levels.length < best.levels)
      if (better) best = { score, levels: levels.length, detection: { kind, strip } }
    }
  }
  // Nothing explained any folder: fall back to "no folder meaning" instead of the first candidate.
  return best && best.score > 0 && best.levels > 0 ? best.detection : { kind: 'flat', strip: 0 }
}

export interface Placement {
  /** Folder name read from the path, or `null` when the layout has no class level (or the path lacks one). */
  className: string | null
  split: SplitType | null
  /** Folders left below the mapped levels. */
  extraDepth: number
}

/** Reads a file's class/split out of its directory path under `kind`. Levels that don't match are simply absent. */
export function placeEntry(dirs: string[], kind: LayoutKind, strip: number): Placement {
  const rest = dirs.slice(strip)
  const placement: Placement = { className: null, split: null, extraDepth: 0 }
  let n = 0
  for (const level of LAYOUT_LEVELS[kind]) {
    if (!fitsLevel(rest[n], level)) break
    if (level === 'class') placement.className = rest[n]
    else placement.split = splitFromName(rest[n])
    n++
  }
  placement.extraDepth = rest.length - n
  return placement
}
