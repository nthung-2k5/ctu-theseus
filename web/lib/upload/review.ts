/**
 * The review table's rows: one shape for everything that can be staged, files and inline items alike, plus the
 * filters on top of it. The table never knows which kind it is showing; `fileToRow` / `itemToRow` do the translating
 * and the staging hooks translate edits back.
 */

import type { StagedItem } from '../uploadQueue'
import { existingKey, fileIssues, hasBlockingError, NO_CLASS_MESSAGE } from './fileTree'
import type { Issue, StagedFile, StagedSplit } from './types'

export interface ReviewRow {
  id: string
  name: string
  /** A dimmed second line: character count, feature summary. */
  detail?: string
  size?: number
  /** What kind of file this is, for its icon; absent for an inline row. */
  media?: 'image' | 'audio' | 'video' | 'file'
  /** The file itself when it can be shown as a thumbnail (a readable image). */
  thumb?: File
  split: StagedSplit
  /** `c:<classId>` for an existing class, `n:<name>` for one to be created, `null` for none. */
  classKey: string | null
  /** The task labels with classes and this row has none. */
  unclassified: boolean
  issues: Issue[]
  /** Why the row is not sent (its content is already there). */
  skip?: string
}

export type ReviewFilter = 'all' | 'errors' | 'warnings' | 'unclassified' | 'skipped' | 'auto'

export const FILTER_LABELS: Record<ReviewFilter, string> = {
  all: 'All',
  errors: 'Errors',
  warnings: 'Warnings',
  unclassified: 'No class',
  skipped: 'Already added',
  auto: 'Auto-split',
}

export function mediaOf(mime: string): NonNullable<ReviewRow['media']> {
  if (mime.startsWith('image/')) return 'image'
  if (mime.startsWith('audio/')) return 'audio'
  if (mime.startsWith('video/')) return 'video'
  return 'file'
}

export function fileToRow(file: StagedFile, taskUsesClasses: boolean): ReviewRow {
  // The no-class warning and the skip reason have their own filters and columns; the rest are the file's problems.
  const issues = fileIssues(file, taskUsesClasses).filter(
    (i) => i.message !== NO_CLASS_MESSAGE && i.message !== skipMessage(file),
  )
  return {
    id: file.id,
    name: file.name,
    size: file.size,
    media: mediaOf(file.file.type),
    // Only an image that will be sent is worth decoding; a broken one is better shown as broken.
    thumb: file.file.type.startsWith('image/') && !hasBlockingError(file) ? file.file : undefined,
    split: file.split,
    classKey: file.classKey,
    unclassified: taskUsesClasses && file.classKey === null,
    issues,
    skip: file.skip,
  }
}

const skipMessage = (file: StagedFile) => (file.skip ? `${file.skip}, so it is not sent` : undefined)

export function itemToRow(item: StagedItem, taskHasClasses: boolean): ReviewRow {
  return {
    id: item.id,
    name: item.name,
    detail: item.detail,
    split: item.split,
    classKey: item.classId ? existingKey(item.classId) : null,
    unclassified: taskHasClasses && !item.classId && item.targetValue == null,
    issues: [],
  }
}

const hasError = (row: ReviewRow) => row.issues.some((i) => i.severity === 'error')
const hasWarning = (row: ReviewRow) => row.issues.some((i) => i.severity === 'warning')

const MATCHES: Record<ReviewFilter, (row: ReviewRow) => boolean> = {
  all: () => true,
  errors: hasError,
  warnings: hasWarning,
  unclassified: (row) => row.unclassified,
  skipped: (row) => row.skip !== undefined && !hasError(row),
  auto: (row) => row.split === 'auto',
}

export function filterRows(rows: ReviewRow[], filter: ReviewFilter): ReviewRow[] {
  return filter === 'all' ? rows : rows.filter(MATCHES[filter])
}

export function countRows(rows: ReviewRow[]): Record<ReviewFilter, number> {
  const counts: Record<ReviewFilter, number> = {
    all: rows.length,
    errors: 0,
    warnings: 0,
    unclassified: 0,
    skipped: 0,
    auto: 0,
  }
  for (const row of rows) {
    for (const filter of Object.keys(MATCHES) as ReviewFilter[]) {
      if (filter !== 'all' && MATCHES[filter](row)) counts[filter]++
    }
  }
  return counts
}

/** The one line a row shows under its name: its most serious problem, else why it is not sent. */
export function rowNote(row: ReviewRow): { text: string; tone: 'error' | 'warning' | 'muted' } | null {
  const error = row.issues.find((i) => i.severity === 'error')
  if (error) return { text: error.message, tone: 'error' }
  const warning = row.issues.find((i) => i.severity === 'warning')
  if (warning) return { text: warning.message, tone: 'warning' }
  return row.skip ? { text: `${row.skip}, so it is not sent`, tone: 'muted' } : null
}
