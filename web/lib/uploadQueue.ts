/**
 * The staging queue behind the Upload page's preview pane for text and tabular tasks.
 *
 * Nothing here talks to the server: the left-hand panels turn typed text and
 * parsed CSV rows into `StagedItem`s, the preview pane lets the user retarget
 * each one's split/class, and only then does `UploadQueuePanel` send the batch.
 * State is in-memory on purpose, so a half-staged queue never looks restorable.
 *
 * File tasks (vision/audio) don't use this: their staging lives in
 * `lib/upload/`, behind the filesystem view.
 */

import type { SplitType } from '@public/store/types'
import { useCallback, useState } from 'react'

export interface StagedItem {
  id: string
  /** Which left-hand panel produced this. A queue is homogeneous — the task's `itemSpec.payload` picks exactly one panel. */
  kind: 'text' | 'csv'
  /** Primary label in the preview table: text excerpt or row number. */
  name: string
  /** Secondary dimmed line: character count, feature summary. */
  detail?: string
  /** Originating file name, when it differs from `name` (CSV rows). Feeds the upload-history entry. */
  sourceName?: string
  split: SplitType
  /** `null` = no class assigned yet. Only meaningful when the task has label classes. */
  classId: string | null
  /** kind: 'text' */
  text?: string
  /** kind: 'csv' */
  featuresJson?: Record<string, string | number>
  /** kind: 'csv' on a regression task — the numeric target, which has no class to assign. */
  targetValue?: number
}

/** What the entry panels hand over; the queue assigns the id. */
export type StagedDraft = Omit<StagedItem, 'id'>

/** The per-row fields the preview pane can edit. */
export type StagedEdit = Partial<Pick<StagedItem, 'split' | 'classId'>>

export function useUploadQueue() {
  const [items, setItems] = useState<StagedItem[]>([])

  const stage = useCallback((drafts: StagedDraft[]) => {
    setItems((prev) => [...prev, ...drafts.map((draft) => ({ ...draft, id: crypto.randomUUID() }))])
  }, [])

  const edit = useCallback((id: string, patch: StagedEdit) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }, [])

  const editAll = useCallback((patch: StagedEdit) => {
    setItems((prev) => prev.map((item) => ({ ...item, ...patch })))
  }, [])

  const remove = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id))
  }, [])

  const clear = useCallback(() => setItems([]), [])

  return { items, stage, edit, editAll, remove, clear }
}
