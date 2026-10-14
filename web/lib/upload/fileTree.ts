/**
 * The staging state behind the add-data panel's filesystem view — pure, no React.
 *
 * Files are staged in a normalized `Map` and tagged with a class *key* and a split; the folder tree the user
 * sees is derived from that (`buildIndex`), never stored. Class keys are `c:<classId>` for an existing class,
 * `n:<normalized name>` for a folder that will become a new class on upload, and `null` for "no class".
 */

import type { SplitType } from '../../store/types'
import type { Skipped } from './archive'
import { candidateLayouts, detectLayout, placeEntry } from './layout'
import { normalizeClassName } from './paths'
import {
  AUTO_SPLIT,
  type DropContext,
  type Issue,
  type LayoutKind,
  type PendingFolder,
  type RawEntry,
  type StagedFile,
  type StagedSplit,
} from './types'

export const SPLITS: SplitType[] = ['train', 'validation', 'test']
/** Every split a staged file can sit in; `auto` last, as the bucket for files nobody has placed yet. */
export const STAGED_SPLITS: StagedSplit[] = [...SPLITS, AUTO_SPLIT]
/** Tree key for files with no class, and for the single root folder of tasks without classes. */
export const NO_CLASS_KEY = '__none__'
export const NO_CLASS_MESSAGE = 'No class assigned'

export interface ClassRef {
  classId: string
  name: string
}

export interface BatchInfo {
  id: number
  kind: LayoutKind
  strip: number
  drop: DropContext
  /** Layouts the user may switch this batch to. */
  candidates: LayoutKind[]
  count: number
}

export interface TreeState {
  files: Map<string, StagedFile>
  pending: Map<string, PendingFolder>
  lastBatch: BatchInfo | null
  nextBatchId: number
  /** OS junk silently left behind, over all drops since the last clear. */
  junk: number
  /** Entries refused at ingest (unsafe path, oversized archive member, unreadable archive...). */
  skipped: Skipped[]
}

export function initialState(): TreeState {
  return { files: new Map(), pending: new Map(), lastBatch: null, nextBatchId: 1, junk: 0, skipped: [] }
}

/* ── Class keys ── */

export const existingKey = (classId: string) => `c:${classId}`
export const pendingKey = (name: string) => `n:${normalizeClassName(name)}`
export const isPendingKey = (key: string | null): key is string => key?.startsWith('n:') ?? false
export const classIdOfKey = (key: string | null): string | null => (key?.startsWith('c:') ? key.slice(2) : null)

/** Finds the key a folder name maps to: an existing class if the name matches one, otherwise a new pending one. */
function resolveClassKey(name: string, classes: ClassRef[], pending: Map<string, PendingFolder>): string {
  const norm = normalizeClassName(name)
  const match = classes.find((c) => normalizeClassName(c.name) === norm)
  if (match) return existingKey(match.classId)
  const key = pendingKey(name)
  if (!pending.has(key)) pending.set(key, { key, name: name.trim() })
  return key
}

/* ── Staging a batch ── */

export interface StageOptions {
  drop: DropContext
  taskUsesClasses: boolean
  classes: ClassRef[]
}

interface Placed {
  classKey: string | null
  split: StagedSplit
  extraDepth: number
}

function place(
  dirs: string[],
  kind: LayoutKind,
  strip: number,
  { drop, taskUsesClasses, classes }: StageOptions,
  pending: Map<string, PendingFolder>,
): Placed {
  const p = placeEntry(dirs, kind, strip)
  const classKey =
    drop.classKey !== undefined
      ? drop.classKey
      : taskUsesClasses && p.className
        ? resolveClassKey(p.className, classes, pending)
        : null
  return { classKey, split: drop.split ?? p.split ?? AUTO_SPLIT, extraDepth: p.extraDepth }
}

export interface StagedBatch {
  files: StagedFile[]
  pending: PendingFolder[]
  batch: BatchInfo
}

/** Detects the layout for a set of ingested entries and turns them into staged files. `issues` lines up with `raw`. */
export function stageBatch(raw: RawEntry[], issues: Issue[][], batchId: number, options: StageOptions): StagedBatch {
  const candidates = candidateLayouts(options.taskUsesClasses, options.drop)
  const dirsList = raw.map((entry) => entry.path.slice(0, -1))
  const { kind, strip } = detectLayout(dirsList, candidates)

  const pending = new Map<string, PendingFolder>()
  const files = raw.map((entry, i): StagedFile => {
    const placed = place(dirsList[i], kind, strip, options, pending)
    return {
      id: crypto.randomUUID(),
      batchId,
      file: entry.file,
      name: entry.file.name,
      size: entry.file.size,
      sourceDirs: dirsList[i],
      issues: issues[i] ?? [],
      ...placed,
    }
  })
  return {
    files,
    pending: [...pending.values()],
    batch: { id: batchId, kind, strip, drop: options.drop, candidates, count: files.length },
  }
}

/* ── Reducer ── */

export type TreeAction =
  | { type: 'add'; staged: StagedBatch; junk: number; skipped: Skipped[] }
  | { type: 'relayout'; kind: LayoutKind; options: StageOptions }
  | { type: 'move'; ids: string[]; classKey?: string | null; split?: StagedSplit }
  | { type: 'remove'; ids: string[] }
  | { type: 'renameFolder'; key: string; name: string; classes: ClassRef[] }
  | { type: 'mapFolder'; from: string; to: string | null }
  | { type: 'reconcile'; classes: ClassRef[] }
  | { type: 'hashed'; hashes: Record<string, string> }
  | { type: 'skip'; reasons: Record<string, string> }
  | { type: 'clear' }

/** Drops pending folders that no file points at any more. */
function gcPending(state: TreeState): TreeState {
  if (state.pending.size === 0) return state
  const used = new Set<string>()
  for (const file of state.files.values()) if (isPendingKey(file.classKey)) used.add(file.classKey)
  if ([...state.pending.keys()].every((key) => used.has(key))) return state
  return { ...state, pending: new Map([...state.pending].filter(([key]) => used.has(key))) }
}

function retarget(files: Map<string, StagedFile>, from: string, to: string | null): Map<string, StagedFile> {
  const next = new Map(files)
  for (const [id, file] of files) if (file.classKey === from) next.set(id, { ...file, classKey: to })
  return next
}

export function treeReducer(state: TreeState, action: TreeAction): TreeState {
  switch (action.type) {
    case 'add': {
      const files = new Map(state.files)
      for (const file of action.staged.files) files.set(file.id, file)
      const pending = new Map(state.pending)
      for (const folder of action.staged.pending) if (!pending.has(folder.key)) pending.set(folder.key, folder)
      return {
        files,
        pending,
        lastBatch: action.staged.batch,
        nextBatchId: state.nextBatchId + 1,
        junk: state.junk + action.junk,
        skipped: [...state.skipped, ...action.skipped],
      }
    }

    case 'relayout': {
      const batch = state.lastBatch
      if (!batch) return state
      const pending = new Map(state.pending)
      const files = new Map(state.files)
      for (const [id, file] of state.files) {
        if (file.batchId !== batch.id) continue
        files.set(id, { ...file, ...place(file.sourceDirs, action.kind, batch.strip, action.options, pending) })
      }
      return gcPending({ ...state, files, pending, lastBatch: { ...batch, kind: action.kind } })
    }

    case 'move': {
      const ids = new Set(action.ids)
      const files = new Map(state.files)
      for (const id of ids) {
        const file = state.files.get(id)
        if (!file) continue
        files.set(id, {
          ...file,
          classKey: action.classKey !== undefined ? action.classKey : file.classKey,
          split: action.split ?? file.split,
        })
      }
      return gcPending({ ...state, files })
    }

    case 'remove': {
      const files = new Map(state.files)
      for (const id of action.ids) files.delete(id)
      return gcPending({ ...state, files })
    }

    case 'renameFolder': {
      const name = action.name.trim()
      if (!name) return state
      const pending = new Map(state.pending)
      const to = resolveClassKey(name, action.classes, pending)
      if (to === action.key) {
        pending.set(to, { key: to, name })
        return { ...state, pending }
      }
      return gcPending({ ...state, files: retarget(state.files, action.key, to), pending })
    }

    case 'mapFolder':
      return gcPending({ ...state, files: retarget(state.files, action.from, action.to) })

    case 'reconcile': {
      let files = state.files
      for (const folder of state.pending.values()) {
        const match = action.classes.find((c) => normalizeClassName(c.name) === normalizeClassName(folder.name))
        if (match) files = retarget(files, folder.key, existingKey(match.classId))
      }
      return files === state.files ? state : gcPending({ ...state, files })
    }

    case 'hashed': {
      const files = new Map(state.files)
      for (const [id, hash] of Object.entries(action.hashes)) {
        const file = state.files.get(id)
        if (file) files.set(id, { ...file, hash })
      }
      return { ...state, files }
    }

    case 'skip': {
      // The reasons cover every hashed file, so a file that is no longer a duplicate (its twin was removed) is cleared.
      let files: Map<string, StagedFile> | null = null
      for (const [id, file] of state.files) {
        const skip = action.reasons[id]
        if (file.skip === skip) continue
        files ??= new Map(state.files)
        files.set(id, { ...file, skip })
      }
      return files ? { ...state, files } : state
    }

    case 'clear':
      return initialState()
  }
}

/* ── Derived view ── */

/** Everything the UI says about one file: its own issues, plus placement and server problems. */
export function fileIssues(file: StagedFile, taskUsesClasses: boolean): Issue[] {
  const issues = [...file.issues]
  if (file.extraDepth > 0) {
    issues.push({
      severity: 'warning',
      message: `${file.extraDepth} deeper folder level${file.extraDepth === 1 ? '' : 's'} ignored`,
    })
  }
  if (taskUsesClasses && file.classKey === null) issues.push({ severity: 'warning', message: NO_CLASS_MESSAGE })
  if (file.skip) issues.push({ severity: 'warning', message: `${file.skip}, so it is not sent` })
  return issues
}

/** A file that can never upload as it is (wrong type, too big, empty). */
export const hasBlockingError = (file: StagedFile): boolean => file.issues.some((i) => i.severity === 'error')

/** Files with an error stay out of an upload, and so do files whose content is already there. */
export const isUploadable = (file: StagedFile): boolean => !hasBlockingError(file) && !file.skip

export interface FolderNode {
  key: string
  name: string
  /** `null` for a folder that will be created as a new class, and for the no-class / root folder. */
  classId: string | null
  pending: boolean
  /** The no-class folder, or the single root folder of a task without classes. */
  isNone: boolean
  files: Record<StagedSplit, StagedFile[]>
  total: number
}

export interface Totals {
  files: number
  uploadable: number
  withErrors: number
  withWarnings: number
  unclassified: number
  /** Names of folders that will be created as classes: pending folders with at least one file that will upload. */
  newClassNames: string[]
  /** Files left out because the dataset already has their content (or an earlier staged file does). */
  skipped: number
  bySplit: Record<StagedSplit, number>
}

export interface TreeIndex {
  folders: FolderNode[]
  totals: Totals
}

const emptySplits = (): Record<StagedSplit, StagedFile[]> => ({ train: [], validation: [], test: [], auto: [] })

function newNode(key: string, name: string, extra: Partial<FolderNode> = {}): FolderNode {
  return { key, name, classId: null, pending: false, isNone: false, files: emptySplits(), total: 0, ...extra }
}

/** Groups staged files under their class folders. Existing classes always appear, even when empty. */
export function buildIndex(state: TreeState, classes: ClassRef[], taskUsesClasses: boolean): TreeIndex {
  const nodes = new Map<string, FolderNode>()
  if (taskUsesClasses) {
    for (const c of [...classes].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))) {
      nodes.set(existingKey(c.classId), newNode(existingKey(c.classId), c.name, { classId: c.classId }))
    }
    for (const p of [...state.pending.values()].sort((a, b) =>
      a.name.localeCompare(b.name, undefined, { numeric: true }),
    )) {
      nodes.set(p.key, newNode(p.key, p.name, { pending: true }))
    }
  }
  const none = newNode(NO_CLASS_KEY, taskUsesClasses ? 'No class' : 'Files', { isNone: true })

  const totals: Totals = {
    files: 0,
    uploadable: 0,
    withErrors: 0,
    withWarnings: 0,
    unclassified: 0,
    newClassNames: [],
    skipped: 0,
    bySplit: { train: 0, validation: 0, test: 0, auto: 0 },
  }

  const needed = new Set<string>()
  for (const file of state.files.values()) {
    const key = taskUsesClasses ? file.classKey : null
    let node = key === null ? none : nodes.get(key)
    if (!node) {
      // A class that vanished (deleted elsewhere) while files were staged against it.
      node = newNode(key as string, 'Unknown class')
      nodes.set(node.key, node)
    }
    node.files[file.split].push(file)
    node.total++

    totals.files++
    totals.bySplit[file.split]++
    if (file.skip && !hasBlockingError(file)) totals.skipped++
    if (isUploadable(file)) {
      totals.uploadable++
      if (isPendingKey(file.classKey)) needed.add(file.classKey)
    }
    if (taskUsesClasses && file.classKey === null) totals.unclassified++
    if (hasBlockingError(file)) totals.withErrors++
    if (file.issues.some((i) => i.severity === 'warning') || file.extraDepth > 0) totals.withWarnings++
  }

  totals.newClassNames = [...needed].flatMap((key) => state.pending.get(key)?.name ?? [])

  const folders = [...nodes.values()]
  if (none.total > 0 || !taskUsesClasses) folders.push(none)
  return { folders, totals }
}
