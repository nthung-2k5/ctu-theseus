/**
 * The Classes page's editable draft, as pure functions so the rules can be unit-tested.
 *
 * The page edits a local list and saves it in one batch (PUT /projects/:id/classes). `rows` is the list as it
 * would be saved; a class that is in the saved list but not in `rows` is being removed, and `mergeInto` can say
 * where its items go instead of being left unlabelled.
 */

/** Categorical palette, reused for new classes (first colour not already taken). */
export const CLASS_PALETTE = [
  '#00afef',
  '#f59e0b',
  '#34d399',
  '#f472b6',
  '#a78bfa',
  '#4a89d1',
  '#fb7185',
  '#22d3ee',
  '#facc15',
  '#86efac',
  '#c084fc',
  '#f97316',
]

export const MAX_NAME = 100
export const MAX_CLASSES = 200
export const FALLBACK_COLOR = '#64748b'

/** One editable row. `classId` is absent for a class that has not been saved yet. */
export interface DraftClass {
  key: string
  classId?: string
  name: string
  description: string
  color: string
}

/** A class as the API returns it (the fields the draft is built from). */
export interface SavedClass {
  classId: string
  name: string
  description?: string | null
  uiColorHex?: string | null
}

export const norm = (s: string) => s.trim().replace(/\s+/g, ' ')
export const fold = (s: string) => norm(s).toLowerCase()

let newKeyCounter = 0
export const newKey = () => `new-${++newKeyCounter}`

export const fromSaved = (c: SavedClass): DraftClass => ({
  key: c.classId,
  classId: c.classId,
  name: c.name,
  description: c.description ?? '',
  color: c.uiColorHex ?? FALLBACK_COLOR,
})

export function nextColor(classes: DraftClass[]): string {
  const used = new Set(classes.map((c) => c.color.toLowerCase()))
  return CLASS_PALETTE.find((c) => !used.has(c)) ?? CLASS_PALETTE[classes.length % CLASS_PALETTE.length]
}

/** Why a name can't be used, or null when it is fine. A row may keep its own name. */
export function nameError(name: string, draft: DraftClass[], ignoreKey?: string): string | null {
  const n = norm(name)
  if (!n) return 'Name is required'
  if (n.length > MAX_NAME) return `At most ${MAX_NAME} characters`
  if (draft.some((c) => c.key !== ignoreKey && fold(c.name) === fold(n))) return 'Already exists'
  return null
}

/** Split pasted text (newlines, commas, semicolons) into new names, reporting what was left out and why. */
export function parseBulk(text: string, draft: DraftClass[]) {
  const seen = new Set(draft.map((c) => fold(c.name)))
  const names: string[] = []
  const skipped: string[] = []
  for (const raw of text.split(/[\n,;]+/)) {
    const n = norm(raw)
    if (!n) continue
    if (n.length > MAX_NAME || seen.has(fold(n))) {
      skipped.push(n)
      continue
    }
    seen.add(fold(n))
    names.push(n)
  }
  return { names, skipped }
}

/** Whether pasted text is a list rather than one name, which is the cue to offer the bulk dialog. */
export const looksLikeList = (text: string) => /[\n,;]/.test(text.trim())

export const sameRow = (a: DraftClass, b: DraftClass) =>
  a.key === b.key &&
  a.name === b.name &&
  a.description === b.description &&
  a.color.toLowerCase() === b.color.toLowerCase()

export const isDirty = (baseline: DraftClass[], rows: DraftClass[]) =>
  rows.length !== baseline.length || rows.some((c, i) => !sameRow(c, baseline[i]))

/** Saved classes that `rows` no longer contains: the ones this save would remove. */
export const removedFrom = (baseline: DraftClass[], rows: DraftClass[]) => {
  const kept = new Set(rows.map((r) => r.key))
  return baseline.filter((c) => !kept.has(c.key))
}

/**
 * Put a restored class back where it was: saved classes keep the saved (creation) order and classes added in this
 * session stay after them, in the order they were added.
 */
export function restoreRow(baseline: DraftClass[], rows: DraftClass[], restored: DraftClass): DraftClass[] {
  const order = new Map(baseline.map((c, i) => [c.key, i]))
  const all = [...rows, restored]
  const saved = all.filter((c) => order.has(c.key)).sort((a, b) => (order.get(a.key) ?? 0) - (order.get(b.key) ?? 0))
  return [...saved, ...all.filter((c) => !order.has(c.key))]
}

export interface Merge {
  fromClassId: string
  intoClassId: string
}

/**
 * The merges to send with a save. `mergeInto` maps a removed class id to the class id its items should move to,
 * but that target may itself have been removed since (A -> B, then B removed with B -> C): follow the chain to a
 * kept class. A removal whose chain ends in a removed class with no target of its own can't be merged, so its
 * items are left unlabelled and it is reported in `unresolved`.
 */
export function resolveMerges(
  removedIds: readonly string[],
  mergeInto: Readonly<Record<string, string>>,
  keptIds: ReadonlySet<string>,
): { merges: Merge[]; unresolved: string[] } {
  const removed = new Set(removedIds)
  const merges: Merge[] = []
  const unresolved: string[] = []
  for (const from of removedIds) {
    if (mergeInto[from] === undefined) continue // never given a target: leave its items unlabelled, as before
    let target: string | undefined = mergeInto[from]
    const seen = new Set([from])
    while (target !== undefined && removed.has(target) && !seen.has(target)) {
      seen.add(target)
      target = mergeInto[target]
    }
    // Ended on a kept class: merge. Ended on a removed one with nowhere further to go (or a cycle): unresolved.
    if (target !== undefined && keptIds.has(target)) merges.push({ fromClassId: from, intoClassId: target })
    else unresolved.push(from)
  }
  return { merges, unresolved }
}

/** What a save would do, for the summary next to the Save button. */
export function summarize(baseline: DraftClass[], rows: DraftClass[], merged: number) {
  const byKey = new Map(baseline.map((c) => [c.key, c]))
  const kept = rows.filter((c) => byKey.has(c.key))
  return {
    added: rows.filter((c) => !byKey.has(c.key)).length,
    removed: removedFrom(baseline, rows).length,
    renamed: kept.filter((c) => c.name !== byKey.get(c.key)?.name).length,
    merged,
  }
}

/** The body of PUT /projects/:id/classes for the current draft. */
export const toSaveBody = (rows: DraftClass[], merges: Merge[]) => ({
  classes: rows.map((c) => ({
    ...(c.classId && { classId: c.classId }),
    name: norm(c.name),
    description: c.description.trim() || null,
    uiColorHex: c.color,
  })),
  merges,
})

/* ── Health of a class, from the server's per-class counts ── */

export const SPLITS = ['train', 'validation', 'test'] as const
export type SplitName = (typeof SPLITS)[number]

export interface ClassCounts {
  total: number
  train: number
  validation: number
  test: number
}

export type Attention = { kind: 'empty' } | { kind: 'few' } | { kind: 'missing-split'; splits: SplitName[] }

/** The splits that hold any labelled item at all. A split nothing is in isn't a problem with any one class. */
export function usedSplits(stats: readonly ClassCounts[]): Set<SplitName> {
  return new Set(SPLITS.filter((s) => stats.some((c) => c[s] > 0)))
}

/**
 * Why a class needs a look, or null. An empty class gives a model nothing to learn; one under `min` items can't
 * be stratified across splits (auto-split skips it); and one that has items but none in a split that the rest of
 * the dataset uses can't be evaluated there. Only the most serious applies, so a 2-item class says "few", not
 * "no validation items" too.
 */
export function attention(
  stat: ClassCounts | undefined,
  min: number,
  splitsInUse: ReadonlySet<SplitName>,
): Attention | null {
  if (!stat) return null
  if (stat.total === 0) return { kind: 'empty' }
  if (stat.total < min) return { kind: 'few' }
  const splits = SPLITS.filter((s) => splitsInUse.has(s) && stat[s] === 0)
  return splits.length > 0 ? { kind: 'missing-split', splits } : null
}

/* ── Large lists: a view over the draft that never changes it ── */

export type SortMode = 'created' | 'name' | 'count' | 'attention'

export function filterRows(rows: DraftClass[], query: string): DraftClass[] {
  const q = fold(query)
  if (!q) return rows
  return rows.filter((r) => fold(r.name).includes(q) || fold(r.description).includes(q))
}

/** Sorts a copy for display. The saved order is creation order and is never changed by viewing. */
export function sortRows(
  rows: DraftClass[],
  mode: SortMode,
  total: (row: DraftClass) => number | undefined,
  needsAttention: (row: DraftClass) => boolean,
): DraftClass[] {
  if (mode === 'created') return rows
  const out = [...rows]
  if (mode === 'name') return out.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
  if (mode === 'count') return out.sort((a, b) => (total(b) ?? -1) - (total(a) ?? -1))
  // Needs attention first, the emptiest of those first; the rest keep creation order (sort is stable).
  return out.sort((a, b) => {
    const wa = needsAttention(a)
    const wb = needsAttention(b)
    if (wa !== wb) return wa ? -1 : 1
    return wa ? (total(a) ?? 0) - (total(b) ?? 0) : 0
  })
}
