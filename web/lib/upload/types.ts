/** Shared shapes for the Upload page's file staging pipeline. */

import type { SplitType } from '../../store/types'

/** One dropped/picked/extracted file with its folder path. `path` is the directory segments plus the file name. */
export interface RawEntry {
  path: string[]
  file: File
}

/** A folder level in a dataset layout. */
export type Level = 'class' | 'split'

/**
 * How folders map onto the dataset: `class/split` = `cat/train/a.jpg`, `split/class` = `train/cat/a.jpg`
 * (ImageFolder style), and so on. `flat` = no meaningful folders.
 */
export type LayoutKind = 'class/split' | 'split/class' | 'class' | 'split' | 'flat'

export interface Issue {
  /** `error` keeps the file out of the upload; `warning` is shown but the file still goes. */
  severity: 'error' | 'warning'
  message: string
}

/** A class folder that doesn't match an existing class yet. Created when the batch is uploaded. */
export interface PendingFolder {
  key: string
  name: string
}

export interface StagedFile {
  id: string
  batchId: number
  file: File
  name: string
  size: number
  /** Folder segments as ingested, before layout mapping; kept so the layout can be re-run. */
  sourceDirs: string[]
  /** `c:<classId>`, `n:<normalized name>` for a class that doesn't exist yet, or `null` for no class. */
  classKey: string | null
  split: SplitType
  /** Folder levels below the mapped ones — the file is kept under the nearest mapped folder. */
  extraDepth: number
  /** Problems with the file itself (type, size, duplicate). Independent of where it is placed. */
  issues: Issue[]
  /** Why the server refused it on the last attempt. */
  serverError?: string
}

/** Where a drop landed. Levels present here are fixed and won't be read from the dropped paths. */
export interface DropContext {
  /** Class folder key; `undefined` = not fixed by the drop target. */
  classKey?: string | null
  split?: SplitType
}
